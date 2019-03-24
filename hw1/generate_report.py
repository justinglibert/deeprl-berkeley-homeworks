from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle
import os

def plot_model_vs_return(title, env_stats):
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    for stat in env_stats:
        ax.plot(stat["returns"], label=stat["model"])
    ax.legend(loc='upper left')
    return fig

def main():
    envs = os.listdir("models")
    env_stats = {}
    for env in envs:
        env_stats[env] = []
        env_dir = os.path.join(os.path.join("models", env))
        models = os.listdir(env_dir)
        for model in models:
            with open(os.path.join(env_dir, model, 'stats.pkl'), 'rb') as f:
                returns = pickle.loads(f.read())
            env_stats[env].append({
                "model": model,
                "returns": returns
            })
    pp = PdfPages('report.pdf')
    for env, stats in env_stats.items():
        fig = plot_model_vs_return(env, stats)
        pp.savefig(fig)
    pp.close()
    

if __name__ == "__main__":
    main()