import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('fivethirtyeight')


def color_define(col_name):
	if "mean" in col_name:
		color = "#30a2da"
	elif "max" in col_name:
		color = "#fc4f30"
	else:
		color = "#6d904f"
	return color


def set_labels(axes, x_label, y_label):
	for a in axes[:, 0]:
		a.set_ylabel(y_label, fontsize=12)

	for a in axes[-1, :]:
		a.set_xlabel(x_label, fontsize=12)


class Plots:

	def __init__(self, path_to_save):

		self.save_path = path_to_save
		os.makedirs(self.save_path, exist_ok=True)

	def draw_plots(self, df, kind="dist"):

		""" Draw plot in accordance with kind
		Parameters:
			df: pd.DataFrame
				DataFrame to plot
			kind: ['dist', 'scatter'], default 'dist'
				kind of plot to create
		"""

		if kind not in ["dist", "scatter"]:
			raise ValueError(f"kind takes options in ['dist', 'scatter'], got {kind}")

		col_to_plot = df.loc[:, "mean":].columns
		fig, axes = plt.subplots(3, 3, figsize=(16, 16))

		if kind == "dist":
			for i, ax in enumerate(axes.flatten()):
				color = color_define(col_to_plot[i])
				sns.histplot(df[col_to_plot[i]].values, kde=True, ax=ax, bins=60, color=color)
				ax.set_title(col_to_plot[i], fontsize=12, fontweight="bold")
				ax.set_ylabel("", fontsize=12)
			set_labels(axes, "", "count")
			plt.suptitle("Feature distribution", fontsize=16, fontweight="bold")

			path = os.path.join(self.save_path, "Distribution.png")
			plt.savefig(path)

		else:
			for i, ax in enumerate(axes.flatten()):
				color = color_define(col_to_plot[i])
				ax.scatter(x=df["gt_corners"], y=df[col_to_plot[i]], color=color)
				ax.set_title(col_to_plot[i], fontsize=12, fontweight="bold")
				ax.set_xticks(df["gt_corners"].unique())
			set_labels(axes, "gt_corners", "value")
			plt.suptitle("Feature distribution by targets", fontsize=16, fontweight="bold")

			path = os.path.join(self.save_path, "Distribution_by_target.png")
			plt.savefig(path)

		plt.tight_layout()
		plt.show()
		return path

	def plot_paths(self):
		paths = [os.path.join(self.save_path, i) for i in os.listdir(self.save_path)]
		return paths