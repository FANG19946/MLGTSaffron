import pandas as pd

df = pd.read_csv("results/KHAN_search_times_imagenet_khan.csv")

print("Average of full CSV:")
print(df.mean(numeric_only=True))