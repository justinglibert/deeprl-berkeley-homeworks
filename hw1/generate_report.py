from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle
import os

def plot_model_vs_return(title, bc_stats, dagger_stats):
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    for stat in bc_stats:
        ax.plot(stat["returns"], label=stat["model"] + " BC")
    for stat in dagger_stats:
        ax.plot(stat["returns"], label=stat["model"] + " DAgger")
    ax.legend(loc='upper left')
    return fig

def collect_stats_from_folder(folder):
    envs = os.listdir(folder)
    env_stats = {}
    for env in envs:
        env_stats[env] = []
        env_dir = os.path.join(os.path.join(folder, env))
        models = os.listdir(env_dir)
        for model in models:
            with open(os.path.join(env_dir, model, 'stats.pkl'), 'rb') as f:
                returns = pickle.loads(f.read())
            env_stats[env].append({
                "model": model,
                "returns": returns
            })
    return env_stats

def main():
    bc_stats = collect_stats_from_folder("models")
    dagger_stats = collect_stats_from_folder("dagger_models")

    pp = PdfPages('report.pdf')
    for env in bc_stats.keys():
        fig = plot_model_vs_return(env, bc_stats[env], dagger_stats[env])
        pp.savefig(fig)
    pp.close()
    

if __name__ == "__main__":
    main()