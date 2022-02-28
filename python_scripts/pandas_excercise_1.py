import pandas as pd
import seaborn as sns


diamonds = sns.load_dataset("diamonds")
tips = sns.load_dataset("tips")

# Write to the same excel file
with pd.ExcelWriter("data/data.xlsx") as writer:

    diamonds.to_excel(writer, sheet_name="diamonds")
    tips.to_excel(writer, sheet_name="tips")



df_preped = (diamonds.pipe(drop_duplicates).
                      pipe(remove_outliers, ['price', 'carat', 'depth']).
                      pipe(encode_categoricals, ['cut', 'color', 'clarity'])
            )

diamonds["cut_enc"] = pd.factorize(diamonds["cut"])[0]

diamonds["cut_enc"].head(5)

diamonds.describe().T.drop("count", axis=1)\
                 .style.highlight_max(color="darkred")

import numpy as np

diamonds.select_dtypes(include=np.number).head()

diamonds.select_dtypes(exclude=np.number).head()

diamonds.nlargest(5, "price")