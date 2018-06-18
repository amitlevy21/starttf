import matplotlib.pyplot as plt
import os
import tensorflow as tf
try:
    from IPython.display import clear_output
    NO_IPYTHON = False
except ModuleNotFoundError:
    NO_IPYTHON = True


class DefaultLossCallback(tf.train.SessionRunHook):
    def __init__(self, hyper_params, losses, checkpoint_dir, inline_plotting=False, report_storage={}, mode="train"):
        """
        A metric plotter and saver.

        Define your metrics in your loss by using the mode_str + "/*" pattern to add you metrics to the metrics dict
        and this callback will be capable auf automatically plotting them and saving them into a csv file.

        The metrics files can be found in your checkpoint folder.

        :param inline_plotting: This parameter is for jupyter notebook users. This will plot the loss not in a file but inside the notebook.
        """
        self.hyper_params = hyper_params
        self.losses = losses
        self.checkpoint_dir = checkpoint_dir
        self.report_storage = report_storage
        self.mode = mode
        self.inline_plotting = inline_plotting and not NO_IPYTHON
        if mode not in self.report_storage:
            if self.checkpoint_dir.endswith("/eval"):
                self.checkpoint_dir = self.checkpoint_dir[:-5]
            if os.path.exists(self.checkpoint_dir + "/images"):
                for file in os.listdir(self.checkpoint_dir + "/images"):
                    if file.endswith(".csv"):
                        self._read_csv(os.path.join(self.checkpoint_dir + "/images", file)) 
            self.plot()
        if mode not in self.report_storage:
            self.report_storage[mode] = {}
        
    def _read_csv(self, filename):
        title = filename.split("/")[-1][:-4].replace(".", "/")
        with open(filename, "r") as f:
            lines = f.read().splitlines()
            cols = lines[0].split(",")
            for col in cols:
                mode = col.split("/")[0]
                if mode not in self.report_storage:
                    self.report_storage[mode] = {}
                if title not in self.report_storage[mode]:
                    self.report_storage[mode][title] = []
                if "step" not in self.report_storage[mode]:
                    self.report_storage[mode]["step"] = []

            for l_idx, row in enumerate(lines[1:]):
                for i, val in enumerate(row.split(",")):
                    mode = cols[i].split("/")[0]
                    if val == "":
                        continue
                    if cols[i].endswith("x"): # Step handling
                        if len(self.report_storage[mode]["step"]) < l_idx:
                            self.report_storage[mode]["step"].append(int(val))
                    else:
                        self.report_storage[mode][title].append(float(val))
        
    def _compute_mean_per_step(self, mode, k):
        steps = []
        values = []
        n = []
        last_step = -1
        for step, val in zip(self.report_storage[mode]["step"], self.report_storage[mode][k]):
            if step != last_step:
                values.append(0)
                steps.append(step)
                n.append(0)
                last_step = step
            values[-1] += val
            n[-1] += 1
                            
        for i in range(len(values)):
            values[i] = values[i] / float(n[i])

        return steps, values
    
    def plot(self):
        dummy_mode = self.report_storage.keys()[0]
        for k in self.report_storage[dummy_mode].keys():
            if k == "step":
                continue
            data = []
            last_step = -1
            for mode in self.report_storage.keys():
                steps, values = self._compute_mean_per_step(mode, k)
                data.append((mode + "/" + k, steps, values))
            create_plot(k, self.checkpoint_dir, data, self.inline_plotting)

    def after_run(self, run_context, run_values):
        results = run_values.results
        if self.mode == "eval" or results["step"] % self.hyper_params.train.save_checkpoint_steps == 0:
            for k in results.keys():
                if k not in self.report_storage[self.mode]:
                    self.report_storage[self.mode][k] = []
                self.report_storage[self.mode][k].append(results[k])

            if self.inline_plotting and self.mode != "eval":
                clear_output()

            print("{}: Step {}, Loss {}".format(self.mode, self.report_storage[self.mode]["step"][-1], self.report_storage[self.mode]["loss"][-1]))

            if self.mode != "eval":
                self.plot()

    def before_run(self, run_context):
        self.losses["step"] = tf.train.get_global_step()
        run_args = tf.train.SessionRunArgs(fetches=self.losses)
        return run_args

    def callback(self, i_step, metrics, reports, model_path):
        if self.inline_plotting:
            clear_output()
        # Interpret report matching to the report defined in reports by create_model
        print("Iter: %d, Train loss: %.4f, Test loss: %.4f" % (i_step, reports[0], reports[1]))
        self.iter_list.append(i_step)
        for i in range(len(reports)):
            if len(self.report_storage) <= i:
                self.report_storage.append([])
            self.report_storage[i].append(reports[i])

        # A map defining how to render the reports.
        plot_data = {}
        report_id = 0
        for i in range(len(metrics)):
            metric = metrics[i]
            for idx in range(len(metric.values())):
                label = list(metric.keys())[idx]
                report = self.report_storage[report_id]
                report_id += 1
                if label.split("/")[-1] not in plot_data:
                    plot_data[label.split("/")[-1]] = [label.split("/")[-1]]
                plot_data[label.split("/")[-1]].append((label, self.iter_list, report))

        for q in plot_data.values():
            create_plot(q[0], model_path, q[1:], self.inline_plotting)


def create_plot(title, model_path, data, inline_plotting=False):
    if not os.path.exists(model_path + "/images"):
        os.makedirs(model_path + "/images")

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(title)
    csv_dat = {}
    for date in data:
        label, x, y = date
        plt.plot(x, y, label=label)
        csv_dat[label + "/x"] = x
        csv_dat[label + "/y"] = y
    plt.legend()

    with open(model_path + "/images/" + title.replace("/", ".") + ".csv", "w") as f:
        cols = list(csv_dat.keys())
        n_cols = len(cols)
        n_rows = max([len(csv_dat[cols[i]]) for i in range(n_cols)])

        csv = ",".join(cols) + "\n"
        for row in range(n_rows):
            line = []
            for col in range(n_cols):
                value = ""
                if len(csv_dat[cols[col]]) > row:
                    value = csv_dat[cols[col]][row]
                line.append("{}".format(value))
            csv += ",".join(line) + "\n"

        f.write(csv)

    if inline_plotting:
        plt.show()
    else:
        plt.savefig(model_path + "/images/" + title.replace("/", ".") + ".png")
    plt.clf()


def f1_score(true, pred, f1_score_class, tresh=0.5):
    correct_positives = 0
    pred_positives = 0
    true_positives = 0

    for t, p in zip(true, pred):
        if t[f1_score_class] > 0.5 and p[f1_score_class] > tresh:
            correct_positives += 1
        if t[f1_score_class] > 0.5:
            true_positives += 1
        if p[f1_score_class] > tresh:
            pred_positives += 1

    if pred_positives > 0:
        precision = correct_positives / pred_positives
    else:
        precision = 0
    if true_positives > 0:
        recall = correct_positives / true_positives
    else:
        recall = 0
    if precision == 0 and recall == 0:
        return 0, 0, 0

    false_positive = pred_positives - correct_positives
    true_negative = len(true) - true_positives

    fpr = false_positive / true_negative
    tpr = correct_positives / true_positives
    return 2 * precision * recall / (precision + recall), tpr, fpr
