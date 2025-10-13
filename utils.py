import numpy as np

from scipy.stats import norm
from math import sqrt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pingouin import intraclass_corr
import matplotlib.pyplot as plt

import math
def rsquareCI(R2, n, k, CI=0.95):
    """
    From:https://agleontyev.netlify.app/post/2019-09-05-calculating-r-squared-confidence-intervals/
    - R2, r2 of the linear regression model
    - n, number of observations
    - k, number of variable

    # se_lci_uci = metrics_qt_df.sort_values("r2_score").apply(lambda x: rsquareCI(x['r2_score'], x['N'], 1), axis=1)
    # se,lci,uci = se_lci_uci.str[0], se_lci_uci.str[1], se_lci_uci.str[2]

    # metrics_qt_df['SE_R2'] = se
    # metrics_qt_df['LCI_R2'] = lci
    # metrics_qt_df['UCI_R2'] = uci
    # metrics_qt_df
    """
    SE = sqrt((4 * R2 * ((1 - R2) ** 2) * ((n - k - 1) ** 2)) / ((n**2 - 1) * (n + 3)))
    if CI == 0.67:
        upper = R2 + SE
        lower = R2 - SE
    elif CI == 0.8:
        upper = R2 + 1.3 * SE
        lower = R2 - 1.3 * SE
    elif CI == 0.95:
        upper = R2 + 2 * SE
        lower = R2 - 2 * SE
    elif CI == 0.99:
        upper = R2 + 2.6 * SE
        lower = R2 - 2.6 * SE
    else:
        raise ValueError("Unknown value for CI. Please use 0.67, 0.8, 0.95 or 0.99")
    # print("CI:{}\n CI lower boundary:{}\n CI upper boundary:{}".format(CI,lower, upper))
    return SE, lower, upper



def R2difference(R1, R2, SE1, SE2, n1, n2, pooled=True):
    if pooled == False:
        SEdiff = sqrt(SE1**2 + SE2**2)
    elif pooled == True:
        SEdiff = sqrt(((SE1**2) * (n1 - 1) + (SE2**2) * (n2 - 1)) / (n1 + n2 - 2))
    Rdiff = R1 - R2
    z = Rdiff / SEdiff
    p = 2 * (1 - norm.cdf(z))
    # print ("P-value is {}".format(p))
    return (p, z)





def rsquareCI_v2(R2, n, k, CI=0.95):
    R2 = np.asarray(R2, dtype=float)
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)

    SE = np.sqrt((4 * R2 * (1 - R2) ** 2 * (n - k - 1) ** 2) / ((n**2 - 1) * (n + 3)))

    # CI 对应的倍数
    CI_multipliers = {0.67: 1.0, 0.8: 1.3, 0.95: 2, 0.99: 2.6}  # 95% CI

    if CI not in CI_multipliers:
        raise ValueError("Unknown value for CI. Please use 0.67, 0.8, 0.95 or 0.99")

    m = CI_multipliers[CI]
    upper = R2 + m * SE
    lower = R2 - m * SE

    return SE, lower, upper



import re


def get_cat_var_name(x):
    if x.startswith("C("):
        return re.findall(r"\((.*?)\)", x)[0]
    else:
        return x


def get_cat_var_subname(x):
    # match [*]
    try:
        return re.findall(r"\[(.*?)\]", x)[0].split(".")[1]
    except:
        return x


class columnsFormatV1:
    """
    format columns of df to remove space

    use format to remove

    use reverse to get original column name from formatted column name
    """

    def __init__(self, data):
        self.data = data
        self.columns = data.columns

        self.special_chars = (
            "≥≤·！@#￥%……&*（）—+，。？、；：“”‘’《》{}【】 ><+-(),.//%"
        )
        self.columns_dict = {
            i: re.sub(
                f"[{re.escape(self.special_chars)}]",
                "_",
                i,
                # i.translate(str.maketrans("", "", string.punctuation)).replace(
                #     " ", "_"
                # ),
            )
            for i in self.columns
        }
        self.columns_dict_reverse = {v: k for k, v in self.columns_dict.items()}

    def format(self, data):
        return data.rename(columns=self.columns_dict)

    def reverse(self, data):
        return data.rename(columns=self.columns_dict_reverse)

    def get_format_column(self, column):
        if isinstance(column, list):
            return [self.columns_dict.get(i, i) for i in column]
        return self.columns_dict.get(column, column)

    def get_reverse_column(self, column):
        if isinstance(column, list):
            return [self.columns_dict_reverse.get(i, i) for i in column]

        return self.columns_dict_reverse.get(column, column)

    def __str__(self):
        return f"columnsFormat: {self.columns_dict}"

    def __repr__(self):
        return self.__str__()


def calc_relimp(
    data,
    label=None,
    vars=None,
    method="lmg",
    rela=True,
    cat_cols=None,
    binary_y=False,
    formula=None,
):

    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri

    # define the R NULL object
    r_null = robjects.r("NULL")

    # 启用 R 和 pandas 数据转换功能
    pandas2ri.activate()

    # load local package
    # TODO: install R package code
    robjects.r(
        """
        options(warn = -1)
        library(relaimpo)


               """
    )

    calc_relimp_R = robjects.r(
        """

function(
  formula, data,type="lmg",rela=T,cat_cols=NULL, binary_y=F
){
    if (is.null(cat_cols)){
    for (col in cat_cols){
        data[[col]] <- as.factor(data[[col]])
    }
    }




  fit <- glm(formula, data = data)
  crf<-calc.relimp(fit, type = type,rela=rela)
  res <- as.data.frame(
    rbind(
      c(slot(crf,type), slot(crf,"R2.decomp")))
  )
  length <- length(res)
  colnames(res) <- c(colnames(res)[1:length-1], "R2.decomp")

  if (binary_y){
  logit.fit <- glm(formula, data = data, family = binomial(link = "logit"))
  R2 <- 1 - logit.fit$deviance / logit.fit$null.deviance
    res$R2.decomp <- R2
  }
  return (res)
    }
  


        """
    )
    # format the data

    if formula is None and label is not None and vars is not None:
        formula = label + "~" + "+".join(vars)
    elif formula is None:
        pass
    else:
        raise ValueError("Please provide formula or label and vars")
    res = calc_relimp_R(
        formula=formula,
        data=data,
        type=method,
        rela=rela,
        binary_y=robjects.BoolVector([binary_y]),
        cat_cols=robjects.StrVector(cat_cols) if cat_cols else r_null,
    )

    res = pandas2ri.rpy2py(res).reset_index(drop=True)

    return res


# ICC 




def determine_sampling_size(df, test_name_col, quantile=0.5):
    """
    Calculate a reasonable sampling size M based on the distribution of measurements per group (ensuring at least 80% of groups have sample size >= M).

    Parameters:
        - df: pandas.DataFrame, input data.
        - test_name_col: str, column name for the test group.
        - quantile: float, quantile to calculate (default is 0.5, i.e., the median).

    Returns:
        - M: int, recommended sampling size.
    """
    # Count the number of samples per group
    sample_counts = df.groupby(test_name_col).size()

    # Calculate the (1 - quantile) quantile as the recommended M
    M = int(np.quantile(sample_counts, 1 - quantile))
    print(
        f"Recommended sampling size (M) to cover {int((1 - quantile) * 100)}% of data: {M}"
    )
    return M


def compute_pvalue(icc_values, null_hypothesis=0.0):
    """
    Based on the distribution of ICC values obtained by bootstrapping, calculate the p-value of the significance of ICC.

    Parameters:
        - icc_values: list, ICC values obtained by bootstrapping.
        - null_hypothesis: float, the ICC value under the null hypothesis (default 0.0).

    Returns:
        - p_value: float, the p-value of the significance of ICC.
    """
    # Calculate mean and standard deviation
    mean_icc = np.mean(icc_values)
    std_icc = np.std(icc_values)

    # Calculate two-tailed p-value
    z_score = (mean_icc - null_hypothesis) / std_icc
    p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed significance
    return p_value


def sample_icc_optimized(df, test_name_col, prob_col, M=None, k=100, quantile=0.5):
    """
    Randomly sample M samples for each test_name and calculate ICC.

    Parameters:
        - df: pandas.DataFrame, input data.
        - test_name_col: str, column name for the test group.
        - prob_col: str, column name for the probability values.
        - M: int, number of samples per group (if None, the recommended value will be calculated automatically).
        - k: int, number of sampling iterations.
        - quantile: float, quantile used to calculate the recommended M (default is 0.5, i.e., the median).

    Returns:
        - icc_results: dict, contains the mean, standard deviation, confidence interval, and p-value of ICC.
    """

    # If M is None, calculate the recommended M
    if M is None:
        M = determine_sampling_size(df, test_name_col, quantile)

    icc_values_all = []  # Store all sampled ICC results

    for _ in range(k):
        # Create an empty DataFrame to store each sampling result
        sampled_data = []

        # Iterate over each test_name
        for test_name, group in df.groupby(test_name_col):
            # If the group has fewer than M samples, sample with replacement; otherwise, sample without replacement
            sampled_rows = group.sample(n=M, replace=len(group) < M).copy()
            # Add a dummy rater column (to distinguish sampling sources)
            sampled_rows = sampled_rows.reset_index(drop=True).reset_index(
                names="fake_raters"
            )
            sampled_data.append(sampled_rows)

        # Combine sampled data into a new DataFrame
        sampled_data = pd.concat(sampled_data)

        # Use pingouin to calculate ICC
        icc_result = intraclass_corr(
            data=sampled_data,
            targets=test_name_col,  # Grouping column
            raters="fake_raters",  # Data source identifier
            ratings=prob_col,  # Probability values
        )

        # Save results for all ICC types
        icc_values_all.append(icc_result)

    # If only one sampling iteration, return the single result
    if k == 1:
        return icc_values_all[0]

    # For multiple sampling iterations, calculate mean, standard deviation, and confidence interval for each ICC type
    icc_summary = []

    for icc_type in icc_values_all[0]["Type"].unique():
        # Extract ICC values of this type from all sampling iterations
        icc_values = [
            result.loc[result["Type"] == icc_type, "ICC"].values[0]
            for result in icc_values_all
        ]

        # Calculate mean, standard deviation, and 95% confidence interval
        icc_mean = np.mean(icc_values)
        icc_std = np.std(icc_values)
        ci_lower, ci_upper = norm.interval(
            0.95, loc=icc_mean, scale=icc_std / np.sqrt(k)
        )

        # Calculate p-value
        p_value = compute_pvalue(icc_values)

        icc_summary.append(
            {
                "Type": icc_type,
                "Mean ICC": icc_mean,
                "Std ICC": icc_std,
                "95% CI Lower": ci_lower,
                "95% CI Upper": ci_upper,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(icc_summary)


# plot utils

def nice_ticks(min_val, max_val, num_ticks=5):
    """
    生成范围在 [min_val, max_val] 内的 ~num_ticks 个刻度，
    刻度值尽量是 1、2、5 乘以 10 的幂（即 5 和 10 的倍数）。
    """
    if min_val == max_val:
        return [min_val]  # 避免0范围

    raw_range = max_val - min_val
    raw_step = raw_range / (num_ticks - 1)

    # 找到10的幂次
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10**exponent)

    # 优先选择 1, 2, 5 作为步骤的基数
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    step = nice_fraction * (10**exponent)

    # 向下取整到步长倍数
    nice_min = math.floor(min_val / step) * step
    # 向上取整到步长倍数
    nice_max = math.ceil(max_val / step) * step

    # 生成ticks
    ticks = np.arange(nice_min, nice_max + step / 2, step)
    # 保留合理的小数位
    decimal_places = max(0, -exponent)
    ticks = [round(t, decimal_places + 2) for t in ticks]

    return ticks

