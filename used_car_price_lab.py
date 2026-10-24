"""
Used Car Price Lab - interactive ML mini project
================================================
Neural Network (MLP, from scratch) vs Analytical model (Ridge/OLS Normal Equation).

* Needs only Python 3.8+ and numpy  (pip install numpy).  The GUI uses tkinter,
  which ships with standard Python (Windows/macOS). No other files are needed.
* Run:            python used_car_price_lab.py          (opens the GUI)
* Text-mode run:  python used_car_price_lab.py --cli    (no GUI needed)

What you can change inside the program:
  - dataset: size, noise, seed, test split, or load your own CSV
  - analytical model: ridge strength (lambda), polynomial/interaction terms
  - neural network: hidden layers, activation, learning rate, L2, epochs, batch, patience
  - 5-fold cross-validation on/off
  - enter your own car and get both predictions + a price-vs-age what-if curve
  - export dataset, predictions, metrics report

DATA NOTE: the built-in dataset is SIMULATED (realistic depreciation rules + noise),
not real market data. Prices are in INR lakh.
"""
import os, sys, csv, json, queue, threading, time
try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required. Install it with:  pip install numpy")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car_project_output")
FUELS, TRANS, BRANDS = ["Petrol", "Diesel", "CNG"], ["Manual", "Automatic"], ["Economy", "Mid", "Premium"]
COLS = ["age_years", "km_driven", "engine_cc", "fuel", "transmission", "owners", "brand_tier", "price_lakh_inr"]
FEATS = ["age_years", "km_driven", "engine_cc", "owners", "fuel_Diesel", "fuel_CNG",
         "trans_Automatic", "brand_Mid", "brand_Premium"]
DEFAULTS = dict(n=1500, noise=5.0, seed=42, test=20, lam=0.0, poly=1, hidden="64,32", act="relu",
                lr=0.01, l2=1e-4, epochs=400, batch=64, patience=25, cv=False)


# ------------------------------------------------------------------ data
def generate_dataset(n=1500, noise=5.0, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(1, 16, n)
    km = np.clip(rng.normal(11000 * age, 4000, n), 2000, 250000).round()
    eng = rng.choice([800, 1000, 1200, 1500, 1800, 2200], n, p=[.08, .18, .30, .22, .13, .09])
    fuel = rng.choice(FUELS, n, p=[.55, .38, .07]); tr = rng.choice(TRANS, n, p=[.72, .28])
    own = rng.choice([1, 2, 3], n, p=[.70, .24, .06]); br = rng.choice(BRANDS, n, p=[.45, .40, .15])
    base = (2.5 + eng / 1000 * 2.8)
    base = base * np.select([br == "Economy", br == "Mid"], [1.0, 1.45], 2.4)
    base = base * np.where(tr == "Automatic", 1.18, 1.0)
    base = base * np.select([fuel == "Petrol", fuel == "Diesel"], [1.0, 1.10], 0.92)
    kp = 1 - 2.2e-6 * km * np.where(fuel == "Diesel", 1.0, .7) * np.where(br == "Premium", 1.4, 1.0)
    price = base * np.exp(-0.095 * age) * np.clip(kp, .3, 1) * (1 - .06 * (own - 1))
    price = np.clip(price * rng.normal(1, noise / 100, n), .3, None).round(3)
    return [dict(age_years=int(a), km_driven=int(k), engine_cc=int(e), fuel=str(f), transmission=str(t),
                 owners=int(o), brand_tier=str(b), price_lakh_inr=float(p))
            for a, k, e, f, t, o, b, p in zip(age, km, eng, fuel, tr, own, br, price)]


def row_to_x(r):
    return [r["age_years"], r["km_driven"], r["engine_cc"], r["owners"], float(r["fuel"] == "Diesel"),
            float(r["fuel"] == "CNG"), float(r["transmission"] == "Automatic"),
            float(r["brand_tier"] == "Mid"), float(r["brand_tier"] == "Premium")]


def to_xy(rows):
    return np.array([row_to_x(r) for r in rows], float), np.array([r["price_lakh_inr"] for r in rows], float)


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        miss = [c for c in COLS if c not in (rd.fieldnames or [])]
        if miss:
            raise ValueError("CSV is missing columns: " + ", ".join(miss) + "\nRequired: " + ", ".join(COLS))
        for i, r in enumerate(rd, start=2):
            try:
                row = dict(age_years=int(float(r["age_years"])), km_driven=int(float(r["km_driven"])),
                           engine_cc=int(float(r["engine_cc"])), fuel=r["fuel"].strip().title() if r["fuel"].strip().upper() != "CNG" else "CNG",
                           transmission=r["transmission"].strip().title(), owners=int(float(r["owners"])),
                           brand_tier=r["brand_tier"].strip().title(), price_lakh_inr=float(r["price_lakh_inr"]))
            except ValueError:
                raise ValueError(f"Row {i}: a numeric column has a bad value.")
            if row["fuel"] not in FUELS or row["transmission"] not in TRANS or row["brand_tier"] not in BRANDS:
                raise ValueError(f"Row {i}: fuel must be {FUELS}, transmission {TRANS}, brand_tier {BRANDS}.")
            rows.append(row)
    if len(rows) < 50:
        raise ValueError("Need at least 50 rows.")
    return rows


def save_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- models
class AnalyticalModel:
    """Ridge / OLS solved in closed form: theta = (A'A + lam*I)^-1 A'y  (intercept unpenalised).
    poly=2 adds squares and pairwise interaction terms (still solved analytically)."""
    def __init__(self, lam=0.0, poly=1):
        self.lam, self.poly = lam, poly

    def _expand(self, X):
        if self.poly < 2:
            return X
        cols = [X] + [(X[:, i] * X[:, j])[:, None] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
        return np.hstack(cols)

    def fit(self, X, y):
        Z = self._expand(np.asarray(X, float))
        self.mu, self.sd = Z.mean(0), Z.std(0); self.sd[self.sd == 0] = 1
        A = np.column_stack([np.ones(len(Z)), (Z - self.mu) / self.sd])
        P = np.eye(A.shape[1]) * self.lam; P[0, 0] = 0
        self.theta = np.linalg.pinv(A.T @ A + P) @ (A.T @ np.asarray(y, float))
        return self

    def predict(self, X):
        Z = (self._expand(np.asarray(X, float)) - self.mu) / self.sd
        return np.column_stack([np.ones(len(Z)), Z]) @ self.theta


class NeuralNetwork:
    """Multi-layer perceptron, any depth. Mini-batch Adam, L2 decay, early stopping (NumPy)."""
    def __init__(self, hidden=(64, 32), act="relu", lr=0.01, l2=1e-4, epochs=400, batch=64, patience=25, seed=42):
        self.hidden, self.act, self.lr, self.l2 = tuple(hidden), act, lr, l2
        self.epochs, self.batch, self.patience, self.seed = epochs, batch, patience, seed
        self.history = {"train": [], "val": []}

    def _f(self, z):
        return np.maximum(0, z) if self.act == "relu" else np.tanh(z)

    def _df(self, a):  # derivative expressed via the activation output
        return (a > 0).astype(float) if self.act == "relu" else 1 - a * a

    def _forward(self, X):
        A = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = A[-1] @ W + b
            A.append(z if i == len(self.W) - 1 else self._f(z))
        return A

    def fit(self, X, y, progress=None, should_stop=None):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float); y = np.asarray(y, float).reshape(-1, 1)
        self.xm, self.xs = X.mean(0), X.std(0); self.xs[self.xs == 0] = 1
        self.ym, self.ys = y.mean(), (y.std() or 1.0)
        Xs, ys = (X - self.xm) / self.xs, (y - self.ym) / self.ys
        p = rng.permutation(len(Xs)); nv = max(1, int(.1 * len(Xs)))
        Xv, yv, Xt, yt = Xs[p[:nv]], ys[p[:nv]], Xs[p[nv:]], ys[p[nv:]]
        sizes = [Xs.shape[1], *self.hidden, 1]
        sc = (lambda n: np.sqrt(2 / n)) if self.act == "relu" else (lambda n: np.sqrt(1 / n))
        self.W = [rng.normal(0, sc(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((1, b)) for b in sizes[1:]]
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        best, bestp, bad, t = np.inf, None, 0, 0
        self.history = {"train": [], "val": []}
        for ep in range(self.epochs):
            if should_stop and should_stop():
                break
            order = rng.permutation(len(Xt))
            for s in range(0, len(Xt), self.batch):
                i = order[s:s + self.batch]; xb, yb = Xt[i], yt[i]
                A = self._forward(xb)
                d = 2.0 / len(xb) * (A[-1] - yb); gW, gb = [None] * len(self.W), [None] * len(self.W)
                for k in range(len(self.W) - 1, -1, -1):
                    gW[k] = A[k].T @ d + self.l2 * self.W[k]; gb[k] = d.sum(0, keepdims=True)
                    if k > 0:
                        d = (d @ self.W[k].T) * self._df(A[k])
                t += 1
                for k in range(len(self.W)):
                    for P, g, m, v in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                        m[k] = .9 * m[k] + .1 * g[k]; v[k] = .999 * v[k] + .001 * g[k] ** 2
                        P[k] -= self.lr * (m[k] / (1 - .9 ** t)) / (np.sqrt(v[k] / (1 - .999 ** t)) + 1e-8)
            tl = float(np.mean((self._forward(Xt)[-1] - yt) ** 2)); vl = float(np.mean((self._forward(Xv)[-1] - yv) ** 2))
            self.history["train"].append(tl); self.history["val"].append(vl)
            if vl < best - 1e-6:
                best, bad = vl, 0; bestp = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if progress and ep % 5 == 0:
                progress(ep + 1, self.epochs, vl)
        if bestp:
            self.W, self.b = bestp
        self.stopped_epoch = len(self.history["val"])
        return self

    def predict(self, X):
        Xs = (np.asarray(X, float) - self.xm) / self.xs
        return self._forward(Xs)[-1].ravel() * self.ys + self.ym


# --------------------------------------------------------------- metrics
def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def r2(y, p):   return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
def mape(y, p): return float(np.mean(np.abs((y - p) / y)) * 100)
def metrics(y, p): return dict(MAE=mae(y, p), RMSE=rmse(y, p), R2=r2(y, p), MAPE=mape(y, p))


def perm_importance(model, X, y, seed=0, repeats=3):
    rng = np.random.default_rng(seed); base = rmse(y, model.predict(X)); out = []
    for j in range(X.shape[1]):
        d = []
        for _ in range(repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); d.append(rmse(y, model.predict(Xp)) - base)
        out.append(max(0.0, float(np.mean(d))))
    return out


def parse_cfg(c):
    c = dict(c)
    hid = tuple(int(x) for x in str(c["hidden"]).replace(" ", "").split(",") if x)
    if not hid or any(h < 1 or h > 512 for h in hid) or len(hid) > 6:
        raise ValueError("Hidden layers: 1-6 comma-separated sizes between 1 and 512, e.g. 64,32")
    c["hidden"] = hid
    for k, lo, hi in (("n", 100, 20000), ("test", 5, 50), ("epochs", 1, 3000), ("batch", 8, 2048), ("patience", 1, 500)):
        c[k] = int(c[k])
        if not lo <= c[k] <= hi: raise ValueError(f"{k} must be between {lo} and {hi}")
    for k in ("noise", "lam", "lr", "l2"): c[k] = float(c[k])
    if not 0 <= c["noise"] <= 50: raise ValueError("noise % must be 0-50")
    if c["lr"] <= 0 or c["lam"] < 0 or c["l2"] < 0: raise ValueError("lr must be >0; lambda and L2 must be >=0")
    c["poly"] = int(c["poly"]); c["seed"] = int(c["seed"]); c["cv"] = bool(c["cv"])
    if c["act"] not in ("relu", "tanh"): raise ValueError("activation must be relu or tanh")
    return c


def run_pipeline(rows, cfg, log=print, progress=None, should_stop=None):
    cfg = parse_cfg(cfg); X, y = to_xy(rows)
    idx = np.random.default_rng(cfg["seed"]).permutation(len(X)); nt = max(5, int(round(len(X) * cfg["test"] / 100)))
    te, tr = idx[:nt], idx[nt:]
    mk_lin = lambda: AnalyticalModel(cfg["lam"], cfg["poly"])
    mk_nn = lambda: NeuralNetwork(cfg["hidden"], cfg["act"], cfg["lr"], cfg["l2"], cfg["epochs"], cfg["batch"], cfg["patience"], cfg["seed"])
    log(f"Data: {len(tr)} train / {nt} test rows")
    log(f"Training analytical model (Normal Equation, lambda={cfg['lam']}, poly={cfg['poly']})...")
    lin = mk_lin().fit(X[tr], y[tr]); pl = lin.predict(X[te])
    log(f"Training neural network {cfg['hidden']} {cfg['act']}...")
    nn = mk_nn().fit(X[tr], y[tr], progress, should_stop); pn = nn.predict(X[te])
    log(f"Neural network stopped at epoch {nn.stopped_epoch}.")
    res = dict(cfg=cfg, lin=lin, nn=nn, X_te=X[te], y_te=y[te], p_lin=pl, p_nn=pn,
               m_lin=metrics(y[te], pl), m_nn=metrics(y[te], pn), cv=None)
    log("Computing feature importance...")
    res["imp_lin"] = perm_importance(lin, X[te], y[te]); res["imp_nn"] = perm_importance(nn, X[te], y[te])
    if cfg["cv"]:
        k, folds, cvr = 5, np.array_split(np.random.default_rng(cfg["seed"] + 1).permutation(len(X)), 5), {"lin": [], "nn": []}
        for i in range(k):
            if should_stop and should_stop(): break
            log(f"Cross-validation fold {i + 1}/{k}...")
            v = folds[i]; t_ = np.concatenate([folds[j] for j in range(k) if j != i])
            cvr["lin"].append(r2(y[v], mk_lin().fit(X[t_], y[t_]).predict(X[v])))
            cvr["nn"].append(r2(y[v], mk_nn().fit(X[t_], y[t_]).predict(X[v])))
        res["cv"] = cvr
    log("Done.")
    return res


def report_text(res, n_rows):
    c = res["cfg"]; L = ["USED CAR PRICE LAB - REPORT", "(dataset may be simulated; prices in INR lakh)", "",
         f"Rows: {n_rows}   test split: {c['test']}%   seed: {c['seed']}",
         f"Analytical: lambda={c['lam']} poly={c['poly']}   NN: hidden={c['hidden']} act={c['act']} lr={c['lr']} l2={c['l2']}", "",
         f"{'Model':<30}{'MAE':>9}{'RMSE':>9}{'R2':>9}{'MAPE%':>9}"]
    for n, m in (("Analytical (Ridge/OLS)", res["m_lin"]), ("Neural Network (MLP)", res["m_nn"])):
        L.append(f"{n:<30}{m['MAE']:>9.4f}{m['RMSE']:>9.4f}{m['R2']:>9.4f}{m['MAPE']:>9.2f}")
    if res["cv"] and res["cv"]["lin"]:
        L += ["", "5-fold CV R2 (mean +/- std):"]
        for n, k in (("Analytical", "lin"), ("Neural Network", "nn")):
            a = np.array(res["cv"][k]); L.append(f"  {n}: {a.mean():.4f} +/- {a.std():.4f}")
    L += ["", "Feature importance (increase in RMSE when shuffled):", f"{'Feature':<18}{'Analytical':>12}{'NeuralNet':>12}"]
    for f, a, b in zip(FEATS, res["imp_lin"], res["imp_nn"]): L.append(f"{f:<18}{a:>12.4f}{b:>12.4f}")
    return "\n".join(L)


def export_all(rows, res):
    save_csv(os.path.join(OUT_DIR, "dataset.csv"), rows)
    save_csv(os.path.join(OUT_DIR, "predictions.csv"), [dict(Actual=float(a), Analytical=round(float(b), 4), NeuralNet=round(float(c), 4))
             for a, b, c in zip(res["y_te"], res["p_lin"], res["p_nn"])])
    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as f: f.write(report_text(res, len(rows)))
    with open(os.path.join(OUT_DIR, "settings.json"), "w") as f: json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in res["cfg"].items()}, f, indent=2)
    return OUT_DIR


def car_row(age, km, eng, fuel, tr, own, brand):
    return dict(age_years=int(age), km_driven=int(km), engine_cc=int(eng), fuel=fuel, transmission=tr,
                owners=int(own), brand_tier=brand, price_lakh_inr=0.0)


# ------------------------------------------------------------------- CLI
def cli():
    print("=== Used Car Price Lab (text mode) ===\nPress Enter to accept defaults.")
    cfg = dict(DEFAULTS)
    for k, label in (("n", "Number of cars"), ("noise", "Noise %"), ("seed", "Seed"), ("hidden", "NN hidden layers"),
                     ("act", "Activation (relu/tanh)"), ("epochs", "Max epochs"), ("lam", "Ridge lambda"), ("poly", "Polynomial degree (1/2)")):
        try: v = input(f"{label} [{cfg[k]}]: ").strip()
        except EOFError: v = ""
        if v: cfg[k] = v
    cfg["cv"] = (input("Run 5-fold CV? (y/N): ").strip().lower() == "y") if sys.stdin.isatty() else False
    p = parse_cfg(cfg); rows = generate_dataset(p["n"], p["noise"], p["seed"])
    res = run_pipeline(rows, cfg); print("\n" + report_text(res, len(rows)))
    print("\nFiles saved in:", export_all(rows, res))
    print("\nPredict a car (blank age to quit).")
    while sys.stdin.isatty():
        try:
            a = input("Age in years: ").strip()
            if not a: break
            r = car_row(a, input("Km driven: "), input("Engine cc: "), input(f"Fuel {FUELS}: ").strip().title().replace("Cng", "CNG"),
                        input(f"Transmission {TRANS}: ").strip().title(), input("Owners (1-3): "), input(f"Brand tier {BRANDS}: ").strip().title())
            x = np.array([row_to_x(r)], float)
            print(f"  Analytical: {res['lin'].predict(x)[0]:.2f} lakh | Neural net: {res['nn'].predict(x)[0]:.2f} lakh")
        except (ValueError, EOFError) as e:
            print("  Invalid input:", e)


# ------------------------------------------------------------------- GUI
def gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    class App:
        def __init__(s, root):
            s.root, s.rows, s.res, s.q, s.stop, s.busy = root, generate_dataset(), None, queue.Queue(), False, False
            s.vars, s.draws = {}, {}
            root.title("Used Car Price Lab - Neural Network vs Analytical Model"); root.geometry("1100x760")
            nb = ttk.Notebook(root); nb.pack(fill="both", expand=True, padx=6, pady=6); s.nb = nb
            s.t_data, s.t_train, s.t_res, s.t_pred = (ttk.Frame(nb) for _ in range(4))
            for t, n in ((s.t_data, "1. Data"), (s.t_train, "2. Models & Training"), (s.t_res, "3. Results"), (s.t_pred, "4. Predict Your Car")): nb.add(t, text=n)
            s.build_data(); s.build_train(); s.build_res(); s.build_pred(); s.refresh_data(); s.poll()

        def var(s, k, kind=tk.StringVar):
            v = kind(value=DEFAULTS.get(k, "")); s.vars[k] = v; return v

        def field(s, parent, label, key, r, c=0, width=10, values=None):
            ttk.Label(parent, text=label).grid(row=r, column=c, sticky="e", padx=4, pady=3)
            w = (ttk.Combobox(parent, textvariable=s.var(key), values=values, width=width, state="readonly") if values
                 else ttk.Entry(parent, textvariable=s.var(key), width=width))
            w.grid(row=r, column=c + 1, sticky="w", padx=4, pady=3)

        # ---- tab 1
        def build_data(s):
            f = ttk.LabelFrame(s.t_data, text="Dataset settings"); f.pack(fill="x", padx=8, pady=6)
            for i, (l, k) in enumerate((("Cars (100-20000)", "n"), ("Noise % (0-50)", "noise"), ("Seed", "seed"), ("Test split % (5-50)", "test"))):
                s.field(f, l, k, 0, i * 2)
            b = ttk.Frame(f); b.grid(row=1, column=0, columnspan=8, sticky="w", pady=4)
            ttk.Button(b, text="Generate simulated data", command=s.gen).pack(side="left", padx=4)
            ttk.Button(b, text="Load my own CSV...", command=s.load).pack(side="left", padx=4)
            ttk.Button(b, text="Save dataset CSV...", command=s.save_ds).pack(side="left", padx=4)
            s.info = ttk.Label(s.t_data, justify="left"); s.info.pack(anchor="w", padx=10)
            ttk.Label(s.t_data, text="Own CSV needs columns: " + ", ".join(COLS), foreground="#555").pack(anchor="w", padx=10)
            tv = ttk.Treeview(s.t_data, columns=COLS, show="headings", height=18)
            for c in COLS: tv.heading(c, text=c); tv.column(c, width=120, anchor="center")
            tv.pack(fill="both", expand=True, padx=8, pady=6); s.tv = tv

        def gen(s):
            try:
                p = parse_cfg({**s.cfg_raw()}); s.rows = generate_dataset(p["n"], p["noise"], p["seed"]); s.res = None; s.refresh_data()
            except Exception as e: messagebox.showerror("Invalid setting", str(e))

        def load(s):
            path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
            if path:
                try: s.rows = load_csv(path); s.res = None; s.refresh_data()
                except Exception as e: messagebox.showerror("Cannot load CSV", str(e))

        def save_ds(s):
            path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="dataset.csv")
            if path: save_csv(path, s.rows); messagebox.showinfo("Saved", path)

        def refresh_data(s):
            p = np.array([r["price_lakh_inr"] for r in s.rows])
            s.info.config(text=f"{len(s.rows)} cars | price min {p.min():.2f} / mean {p.mean():.2f} / max {p.max():.2f} lakh INR")
            s.tv.delete(*s.tv.get_children())
            for r in s.rows[:200]: s.tv.insert("", "end", values=[r[c] for c in COLS])

        # ---- tab 2
        def build_train(s):
            a = ttk.LabelFrame(s.t_train, text="Analytical model (closed-form)"); a.pack(fill="x", padx=8, pady=6)
            s.field(a, "Ridge lambda (0 = plain OLS)", "lam", 0, 0); s.field(a, "Terms", "poly", 0, 2, values=["1", "2"])
            ttk.Label(a, text="1 = linear, 2 = squares + interactions", foreground="#555").grid(row=0, column=4, padx=6)
            n = ttk.LabelFrame(s.t_train, text="Neural network"); n.pack(fill="x", padx=8, pady=6)
            s.field(n, "Hidden layers", "hidden", 0, 0, 14); s.field(n, "Activation", "act", 0, 2, values=["relu", "tanh"])
            s.field(n, "Learning rate", "lr", 1, 0); s.field(n, "L2 decay", "l2", 1, 2)
            s.field(n, "Max epochs", "epochs", 2, 0); s.field(n, "Batch size", "batch", 2, 2); s.field(n, "Patience", "patience", 2, 4)
            cv = s.var("cv", tk.BooleanVar); cv.set(False)
            ttk.Checkbutton(s.t_train, text="Also run 5-fold cross-validation (slower)", variable=cv).pack(anchor="w", padx=12)
            b = ttk.Frame(s.t_train); b.pack(fill="x", padx=8, pady=6)
            s.btn = ttk.Button(b, text="Train both models", command=s.train); s.btn.pack(side="left", padx=4)
            ttk.Button(b, text="Stop", command=lambda: setattr(s, "stop", True)).pack(side="left", padx=4)
            ttk.Button(b, text="Reset settings", command=s.reset).pack(side="left", padx=4)
            s.pb = ttk.Progressbar(s.t_train, maximum=100); s.pb.pack(fill="x", padx=12, pady=4)
            s.logw = tk.Text(s.t_train, height=16, state="disabled", bg="#fafafa"); s.logw.pack(fill="both", expand=True, padx=8, pady=6)

        def reset(s):
            for k, v in DEFAULTS.items():
                if k in s.vars: s.vars[k].set(v)

        def cfg_raw(s): return {k: v.get() for k, v in s.vars.items()}
        def log(s, m): s.q.put(("log", m))

        def train(s):
            if s.busy: return
            try: parse_cfg(s.cfg_raw())
            except Exception as e: return messagebox.showerror("Invalid setting", str(e))
            s.busy, s.stop = True, False; s.btn.state(["disabled"]); s.logw.config(state="normal"); s.logw.delete("1.0", "end"); s.logw.config(state="disabled")
            cfg, rows = s.cfg_raw(), list(s.rows)

            def work():
                try:
                    r = run_pipeline(rows, cfg, s.log, lambda e, m, v: s.q.put(("prog", e * 100 / m, f"epoch {e}/{m}  val loss {v:.4f}")), lambda: s.stop)
                    s.q.put(("done", r))
                except Exception as e: s.q.put(("err", str(e)))
            threading.Thread(target=work, daemon=True).start()

        def poll(s):
            try:
                while True:
                    m = s.q.get_nowait()
                    if m[0] == "log": s.logw.config(state="normal"); s.logw.insert("end", m[1] + "\n"); s.logw.see("end"); s.logw.config(state="disabled")
                    elif m[0] == "prog": s.pb["value"] = m[1]; s.logw.config(state="normal"); s.logw.insert("end", "  " + m[2] + "\n"); s.logw.see("end"); s.logw.config(state="disabled")
                    elif m[0] == "done": s.res = m[1]; s.busy = False; s.btn.state(["!disabled"]); s.pb["value"] = 100; s.show_res(); s.nb.select(s.t_res)
                    elif m[0] == "err": s.busy = False; s.btn.state(["!disabled"]); messagebox.showerror("Error", m[1])
            except queue.Empty: pass
            s.root.after(100, s.poll)

        # ---- tab 3
        def build_res(s):
            top = ttk.Frame(s.t_res); top.pack(fill="x", padx=8, pady=4)
            s.mt = ttk.Treeview(top, columns=("m", "MAE", "RMSE", "R2", "MAPE"), show="headings", height=2)
            for c, w in (("m", 240), ("MAE", 90), ("RMSE", 90), ("R2", 90), ("MAPE", 90)): s.mt.heading(c, text={"m": "Model", "MAPE": "MAPE %"}.get(c, c)); s.mt.column(c, width=w, anchor="center")
            s.mt.pack(side="left"); s.cvl = ttk.Label(top, justify="left"); s.cvl.pack(side="left", padx=14)
            ttk.Button(top, text="Export all...", command=s.export).pack(side="right")
            g = ttk.Frame(s.t_res); g.pack(fill="both", expand=True)
            for i in range(2): g.rowconfigure(i, weight=1); g.columnconfigure(i, weight=1)
            s.cv_ = [tk.Canvas(g, bg="white", highlightthickness=1, highlightbackground="#ccc") for _ in range(4)]
            for i, c in enumerate(s.cv_):
                c.grid(row=i // 2, column=i % 2, sticky="nsew", padx=3, pady=3); c.bind("<Configure>", lambda e, c=c: s.redraw(c))

        def redraw(s, c):
            if c in s.draws: s.draws[c]()

        def show_res(s):
            r = s.res; s.mt.delete(*s.mt.get_children())
            for n, m in (("Analytical (Ridge/OLS)", r["m_lin"]), ("Neural Network (MLP)", r["m_nn"])):
                s.mt.insert("", "end", values=(n, f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['R2']:.4f}", f"{m['MAPE']:.2f}"))
            cv = r["cv"]; s.cvl.config(text="" if not cv or not cv["lin"] else "5-fold CV R2:  Analytical %.3f +/- %.3f   |   NN %.3f +/- %.3f" % (
                np.mean(cv["lin"]), np.std(cv["lin"]), np.mean(cv["nn"]), np.std(cv["nn"])))
            y = r["y_te"]; h = r["nn"].history
            s.draws[s.cv_[0]] = lambda: scatter(s.cv_[0], y, r["p_lin"], "Analytical: actual vs predicted", "#2b6cb0")
            s.draws[s.cv_[1]] = lambda: scatter(s.cv_[1], y, r["p_nn"], "Neural net: actual vs predicted", "#2f855a")
            s.draws[s.cv_[2]] = lambda: lines(s.cv_[2], [("train", range(1, len(h["train"]) + 1), h["train"], "#2b6cb0"), ("validation", range(1, len(h["val"]) + 1), h["val"], "#c53030")],
                                              "NN loss curve (scaled MSE)", "epoch")
            s.draws[s.cv_[3]] = lambda: bars(s.cv_[3], FEATS, [r["imp_lin"], r["imp_nn"]], ["Analytical", "Neural net"], ["#2b6cb0", "#2f855a"], "Feature importance (RMSE increase when shuffled)")
            for c in s.cv_: s.redraw(c)

        def export(s):
            if s.res: messagebox.showinfo("Exported", "Saved dataset.csv, predictions.csv, report.txt, settings.json in:\n" + export_all(s.rows, s.res))

        # ---- tab 4
        def build_pred(s):
            f = ttk.LabelFrame(s.t_pred, text="Enter car details"); f.pack(fill="x", padx=8, pady=6)
            d = dict(p_age="5", p_km="55000", p_eng="1500", p_fuel="Petrol", p_tr="Manual", p_own="1", p_br="Mid")
            for k, v in d.items(): s.vars[k] = tk.StringVar(value=v)
            def fld(l, k, r, c, vals=None):
                ttk.Label(f, text=l).grid(row=r, column=c, sticky="e", padx=4, pady=3)
                (ttk.Combobox(f, textvariable=s.vars[k], values=vals, width=11, state="readonly") if vals else ttk.Entry(f, textvariable=s.vars[k], width=12)).grid(row=r, column=c + 1, padx=4)
            fld("Age (years)", "p_age", 0, 0); fld("Km driven", "p_km", 0, 2); fld("Engine cc", "p_eng", 0, 4)
            fld("Fuel", "p_fuel", 1, 0, FUELS); fld("Transmission", "p_tr", 1, 2, TRANS); fld("Owners", "p_own", 1, 4, ["1", "2", "3"]); fld("Brand tier", "p_br", 2, 0, BRANDS)
            ttk.Button(f, text="Predict price", command=s.predict).grid(row=2, column=3, columnspan=2, pady=6)
            s.out = ttk.Label(s.t_pred, font=("Segoe UI", 12, "bold"), justify="left"); s.out.pack(anchor="w", padx=12, pady=6)
            s.wc = tk.Canvas(s.t_pred, bg="white", highlightthickness=1, highlightbackground="#ccc"); s.wc.pack(fill="both", expand=True, padx=8, pady=6)
            s.wc.bind("<Configure>", lambda e: s.redraw(s.wc))

        def predict(s):
            if not s.res: return messagebox.showinfo("Train first", "Train the models on tab 2 first.")
            try:
                g = lambda k: s.vars[k].get()
                age, km, eng, own = float(g("p_age")), float(g("p_km")), float(g("p_eng")), float(g("p_own"))
                if not (0 <= age <= 40 and 0 <= km <= 1_000_000 and 500 <= eng <= 6000): raise ValueError("Use age 0-40, km 0-1,000,000, engine 500-6000 cc.")
                base = car_row(age, km, eng, g("p_fuel"), g("p_tr"), own, g("p_br"))
                x = np.array([row_to_x(base)], float); a, b = s.res["lin"].predict(x)[0], s.res["nn"].predict(x)[0]
                s.out.config(text=f"Analytical model: {a:.2f} lakh INR     Neural network: {b:.2f} lakh INR     (difference {abs(a - b):.2f})")
                ages = list(range(0, 21)); xs = np.array([row_to_x({**base, "age_years": i, "km_driven": km * i / max(age, 1) if age >= 1 else 11000 * i}) for i in ages], float)
                pa, pb_ = s.res["lin"].predict(xs), s.res["nn"].predict(xs)
                s.draws[s.wc] = lambda: lines(s.wc, [("Analytical", ages, pa, "#2b6cb0"), ("Neural net", ages, pb_, "#2f855a")],
                                              "What-if: predicted price as the same car ages (km scaled with age)", "age (years)")
                s.redraw(s.wc)
            except ValueError as e: messagebox.showerror("Invalid input", str(e))

    # --- drawing helpers (plain tkinter canvas, no matplotlib needed)
    def frame(c, title, xl=None):
        c.delete("all"); w, h = max(c.winfo_width(), 200), max(c.winfo_height(), 150)
        c.create_text(w / 2, 12, text=title, font=("Segoe UI", 10, "bold")); return w, h, 52, 28, w - 14, h - 38

    def ticks(lo, hi, n=5): return [lo + (hi - lo) * i / n for i in range(n + 1)]

    def scatter(c, y, p, title, color):
        w, h, l, t, r, b = frame(c, title); lo, hi = min(0, float(np.min(p))), max(float(np.max(y)), float(np.max(p))) * 1.03
        X = lambda v: l + (v - lo) / (hi - lo) * (r - l); Y = lambda v: b - (v - lo) / (hi - lo) * (b - t)
        c.create_rectangle(l, t, r, b, outline="#bbb")
        for v in ticks(lo, hi): c.create_text(X(v), b + 10, text=f"{v:.1f}", font=("Segoe UI", 8)); c.create_text(l - 18, Y(v), text=f"{v:.1f}", font=("Segoe UI", 8))
        for a, q in zip(y, p): c.create_oval(X(a) - 2, Y(q) - 2, X(a) + 2, Y(q) + 2, fill=color, outline="")
        c.create_line(X(lo), Y(lo), X(hi), Y(hi), fill="#c53030", dash=(5, 3))
        c.create_text((l + r) / 2, h - 8, text="actual (INR lakh)  -  dashed line = perfect prediction", font=("Segoe UI", 8))

    def lines(c, series, title, xl):
        w, h, l, t, r, b = frame(c, title); xs = [float(v) for s_ in series for v in s_[1]]; ys = [float(v) for s_ in series for v in s_[2]]
        x0, x1, y0, y1 = min(xs), max(xs), min(0, min(ys)), max(ys) * 1.05 or 1
        X = lambda v: l + (v - x0) / ((x1 - x0) or 1) * (r - l); Y = lambda v: b - (v - y0) / (y1 - y0) * (b - t)
        c.create_rectangle(l, t, r, b, outline="#bbb")
        for v in ticks(y0, y1): c.create_text(l - 20, Y(v), text=f"{v:.2f}", font=("Segoe UI", 8))
        for v in ticks(x0, x1): c.create_text(X(v), b + 10, text=f"{v:.0f}", font=("Segoe UI", 8))
        for i, (name, xx, yy, col) in enumerate(series):
            pts = [q for a, v in zip(xx, yy) for q in (X(float(a)), Y(float(v)))]
            if len(pts) >= 4: c.create_line(*pts, fill=col, width=2)
            c.create_text(r - 60, t + 12 + 14 * i, text="- " + name, fill=col, font=("Segoe UI", 9, "bold"))
        c.create_text((l + r) / 2, h - 8, text=xl, font=("Segoe UI", 8))

    def bars(c, names, sets, labels, cols, title):
        w, h, l, t, r, b = frame(c, title); hi = max(max(s_) for s_ in sets) * 1.1 or 1; n = len(names); bw = (r - l) / n
        for i, nm in enumerate(names):
            for k, s_ in enumerate(sets):
                x0 = l + i * bw + 3 + k * (bw - 6) / len(sets); x1 = x0 + (bw - 6) / len(sets) - 1
                c.create_rectangle(x0, b - s_[i] / hi * (b - t), x1, b, fill=cols[k], outline="")
            c.create_text(l + i * bw + bw / 2, b + 12, text=nm.replace("_", "\n", 1), font=("Segoe UI", 7))
        for k, lab in enumerate(labels): c.create_text(r - 60, t + 8 + 13 * k, text="- " + lab, fill=cols[k], font=("Segoe UI", 9, "bold"))

    root = tk.Tk(); App(root); root.mainloop()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        cli()
    else:
        try:
            import tkinter  # noqa
        except ImportError:
            print("tkinter is not available on this Python; starting text mode instead.\n"
                  "(Linux: sudo apt install python3-tk to enable the GUI.)\n")
            cli()
        else:
            gui()
