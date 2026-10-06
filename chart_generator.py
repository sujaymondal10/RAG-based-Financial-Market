import matplotlib.pyplot as plt

def save_price_chart(window, output_path, title=""):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    x = range(len(window))
    ax.plot(x, window["Close"].to_numpy(), linewidth=2, label="Close")
    ax.plot(x, window["Close"].rolling(10, min_periods=1).mean(),
            linewidth=1.5, label="MA10")
    ax.set_title(title)
    ax.set_xlabel("Time window")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
