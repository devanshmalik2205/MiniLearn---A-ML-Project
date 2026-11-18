"""Study tools that need no target column: explore, compare two columns, find groups.
Built on minilearn's helpers, plus pandas and seaborn for advanced EDA."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Required to generate plots headlessly in Flask
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

import minilearn as ml

def _get_base64_image():
    """Helper to convert matplotlib plots to base64 strings for the web."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close('all')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_eda_plots(header, rows, info):
    """Generates professional Seaborn graphs tailored for different ML models."""
    df = pd.DataFrame(rows, columns=header)
    
    # Safely convert only the known numeric columns to numeric (coercing errors to NaN)
    num_cols = [c["name"] for c in info if c["type"] == "numeric"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    plots = {}
    sns.set_theme(style="whitegrid", font_scale=1.1) # Global theme

    if len(num_cols) > 0:
        # 1. Boxplots (Crucial for KNN, SVM, and Linear Models sensitive to outliers)
        plt.figure(figsize=(10, max(4, len(num_cols) * 0.8)))
        ax = sns.boxplot(data=df[num_cols], orient='h', palette="Set3", linewidth=1.5)
        plt.title("Feature Distributions & Outliers (Impacts KNN, SVM, Linear Models)", fontsize=14, pad=15, fontweight='bold')
        plt.xlabel("Value", fontsize=12, fontweight='bold')
        plt.ylabel("Features", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plots['boxplots'] = _get_base64_image()

    if len(num_cols) > 1:
        # 2. Correlation Heatmap (Assumption checking for Linear Regression & Naive Bayes)
        plt.figure(figsize=(max(8, len(num_cols)*0.8), max(6, len(num_cols)*0.6)))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap='RdYlBu_r', fmt=".2f", vmin=-1, vmax=1, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title("Correlation Heatmap (Checks assumptions for Linear Models & Naive Bayes)", fontsize=14, pad=15, fontweight='bold')
        plt.xlabel("Features", fontsize=12, fontweight='bold')
        plt.ylabel("Features", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plots['heatmap'] = _get_base64_image()

        # 3. Violin Plot (Distribution shape for Gaussian Naive Bayes)
        sample_df = df if len(df) <= 400 else df.sample(400, random_state=42)
        if len(num_cols) <= 15: # Prevent rendering issues if too many columns
            plt.figure(figsize=(10, max(4, len(num_cols) * 0.8)))
            melted_df = sample_df[num_cols].melt(var_name='Feature', value_name='Value')
            sns.violinplot(x='Value', y='Feature', data=melted_df, palette="pastel", inner="quartile", linewidth=1.2)
            plt.title("Density & Shape of Features (Vital for Gaussian Naive Bayes)", fontsize=14, pad=15, fontweight='bold')
            plt.xlabel("Value", fontsize=12, fontweight='bold')
            plt.ylabel("Features", fontsize=12, fontweight='bold')
            plt.tight_layout()
            plots['violin'] = _get_base64_image()

        # 4. Pairplot (Reveals complex, non-linear boundaries for Trees & Random Forests)
        cols_to_plot = num_cols[:4] # Limit to 4 features for visual clarity and speed
        if len(cols_to_plot) > 1:
            pp = sns.pairplot(sample_df[cols_to_plot].dropna(), diag_kind='kde', corner=True, 
                              plot_kws={'alpha':0.7, 'edgecolor':'k', 'color': sns.color_palette("husl")[0]},
                              diag_kws={'fill': True, 'color': sns.color_palette("husl")[2]})
            pp.fig.suptitle("Pairwise Relationships (Shows non-linear boundaries for Trees/Forests)", y=1.02, fontsize=14, fontweight='bold')
            for ax in pp.axes.flatten():
                if ax:
                    ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
                    ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
            plots['pairplot'] = _get_base64_image()

    return plots

def _num(rows, i):
    return [float(r[i]) if ml.is_number(r[i]) and not ml.is_missing(r[i]) else np.nan for r in rows]

def _is_id(rows, i):
    present = [r[i] for r in rows if not ml.is_missing(r[i])]
    return bool(present) and ml.looks_like_id(present)

def overview(header, rows, info):
    cols, nums = [], {}
    for i, c in enumerate(info):
        if c["type"] == "numeric" and not _is_id(rows, i):
            raw = _num(rows, i)
            v = np.array(raw)
            v = v[~np.isnan(v)]
            nums[c["name"]] = raw
            cols.append(dict(name=c["name"], kind="numeric", mean=float(v.mean()), median=float(np.median(v)),
                             std=float(v.std()), min=float(v.min()), max=float(v.max()),
                             hist=np.histogram(v, bins=8)[0].tolist()))
        elif c["type"] == "text" and not (c["unique"] > 10 and c["unique"] > 0.5 * len(rows)):
            u, n = np.unique([r[i] for r in rows if not ml.is_missing(r[i])], return_counts=True)
            top = np.argsort(-n)[:5]
            cols.append(dict(name=c["name"], kind="text", top=[[str(u[j]), int(n[j])] for j in top]))
    pairs, names = [], list(nums)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = np.array(nums[names[a]]), np.array(nums[names[b]])
            m = ~np.isnan(x) & ~np.isnan(y)
            if m.sum() > 2 and x[m].std() > 0 and y[m].std() > 0:
                pairs.append([names[a], names[b], float(np.corrcoef(x[m], y[m])[0, 1])])
    pairs.sort(key=lambda p: -abs(p[2]))
    
    # Generate the meaningful graphs
    plots = generate_eda_plots(header, rows, info)
    
    return dict(columns=cols, pairs=pairs[:6], plots=plots)

def compare(header, rows, info, a, b):
    ia, ib = header.index(a), header.index(b)
    ta, tb = info[ia]["type"], info[ib]["type"]
    if ia == ib:
        raise ValueError("Pick two different columns to compare.")
    if ta == "numeric" and tb == "numeric":
        x, y = np.array(_num(rows, ia)), np.array(_num(rows, ib))
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 3 or x[m].std() == 0 or y[m].std() == 0:
            raise ValueError("These columns do not vary enough to compare.")
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        slope = float(np.polyfit(x[m], y[m], 1)[0])
        pts = np.c_[x[m], y[m]][np.random.default_rng(0).permutation(int(m.sum()))[:300]]
        return dict(kind="numeric_numeric", a=a, b=b, n=int(m.sum()), r=r, slope=slope, points=pts.tolist())
    if {ta, tb} == {"numeric", "text"}:
        ni, ci = (ia, ib) if ta == "numeric" else (ib, ia)
        groups = {}
        for r, v in zip(rows, _num(rows, ni)):
            if not np.isnan(v) and not ml.is_missing(r[ci]):
                groups.setdefault(r[ci], []).append(v)
        allv = np.array([v for g in groups.values() for v in g])
        top = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]
        ss_total = float(((allv - allv.mean()) ** 2).sum())
        ss_between = float(sum(len(g) * (np.mean(g) - allv.mean()) ** 2 for g in groups.values()))
        return dict(kind="numeric_text", number=header[ni], group=header[ci],
                    eta2=ss_between / ss_total if ss_total else 0.0,
                    rows=[[k, len(g), float(np.mean(g)), float(np.std(g)), float(np.median(g))] for k, g in top])
    if ta == "text" and tb == "text":
        rl = [k for k, _ in sorted(_count(rows, ia).items(), key=lambda kv: -kv[1])[:6]]
        cl = [k for k, _ in sorted(_count(rows, ib).items(), key=lambda kv: -kv[1])[:6]]
        grid = [[sum(1 for r in rows if r[ia] == p and r[ib] == q) for q in cl] for p in rl]
        return dict(kind="text_text", a=a, b=b, rows=rl, cols=cl, grid=grid)
    raise ValueError("One of these columns is empty.")

def _count(rows, i):
    out = {}
    for r in rows:
        if not ml.is_missing(r[i]):
            out[r[i]] = out.get(r[i], 0) + 1
    return out

def groups(header, rows, info, k):
    idx = []
    for i, c in enumerate(info):
        if c["type"] == "numeric":
            present = [v for v in (r[i] for r in rows) if not ml.is_missing(v)]
            if present and not ml.looks_like_id(present):
                idx.append(i)
    if len(idx) < 2:
        raise ValueError("Finding groups needs at least two numeric columns.")
    X = np.array([_num(rows, i) for i in idx]).T
    X = X[~np.isnan(X).any(axis=1)]
    if len(X) < k * 5:
        raise ValueError("Not enough complete rows for that many groups. Try fewer groups.")
    sd = X.std(axis=0)
    Z = (X - X.mean(axis=0)) / np.where(sd == 0, 1, sd)
    C = Z[np.random.default_rng(0).choice(len(Z), k, replace=False)]
    for _ in range(50):
        lab = ((Z[:, None] - C[None]) ** 2).sum(axis=2).argmin(axis=1)
        C = np.array([Z[lab == j].mean(axis=0) if (lab == j).any() else C[j] for j in range(k)])
    return dict(features=[header[i] for i in idx], rows_used=int(len(X)),
                overall=X.mean(axis=0).tolist(),
                clusters=[dict(size=int((lab == j).sum()), means=X[lab == j].mean(axis=0).tolist())
                          for j in range(k) if (lab == j).any()])