# PG19-group-project-MICROBIOME

# Project Overview

This repository contains code for analyzing and processing datasets related to the microbiome. The code is split into two sections: an Rmd script for performing data analysis and a Python script for machine learning tasks. The necessary datasets are also included in this repository.

## Files and Their Purpose:
**ResearchQuestions1&2.Rmd:** An Rmd script that addresses specific research questions and performs detailed data analysis.

**machinelearning.py:** A Python script that applies machine learning models to the dataset, performing tasks such as model training, evaluation, and prediction.

## Datasets:
The following datasets are required for running the code:

**genera.counts.tsv:** This file contains the count data for different genera.

**metadata.tsv:** This file contains metadata related to the genera, which is necessary for various analyses in both the Rmd and Python scripts.

Download both datasets in raw file.

## How to run the code:
**ResearchQuestion1&2.Rmd:** Download the raw file and open it in RStudio. Specify the location of the downloaded datasets in the code: 
* For Research Question 1, you need to specify the correct file path for the dataset 'metadata' in the file_path variable. Update the file path with the location where the dataset file 'metadata' is stored on your computer, for example: "C:/Users/YourName/Documents".
* Repeat this process when starting Research Question 2 for the genera_counts dataset. Ensure that you provide the correct path for the genera_counts dataset, like so: "C:/Users/YourName/Documents". 

* At the start of the code for Research Question 2, remove the # from the install.packages lines for any libraries you don't already have installed.

Then click 'Run All' to execute the entire script.

**machinelearning.py:** Open a new notebook in Kaggle. As input add the dataset 'metadata.tsv' 
In the right-hand panel of your notebook (next to the code editor), click on the "Add Dataset" button.
A pop-up window will appear showing datasets you've uploaded or ones you've accessed on Kaggle. (previously upload the dataset 'metadata.tsv' in your dataset folder in your kaggle account)
Search for your dataset by name or browse your datasets and select the one you want to use.
Once selected, click on "Add" to link the dataset to your notebook

Then copy the whole code from machinelearning.py and paste it into a cell in your kaggle notebook, then click 'run all' to execute the entire script.

## This file also contains all the codes for all 3 parts (Research Question 1, Research Question 2, and machine learning) with explanation. 

# Research question 1 (univariate analysis): 

## Can calprotectin levels be used as a diagnostic biomarker to differentiate between non-IBD, ulcerative colitis (UC), and Crohn's disease (CD) in patients?

**Null hypothesis:** There is no significant difference in fecal calprotectin levels between patients with CD (Crohn's Disease), UC (Ulcerative Colitis), and non-IBD (individuals without inflammatory bowel diseases).

**Alternative hypothesis:** There is a significant difference in fecal calprotectin levels between patients with CD (Crohn's Disease), UC (Ulcerative Colitis), and non-IBD (individuals without inflammatory bowel diseases).

### Data preparation:

Read data:

```{r}
library(tidyverse)
```

```{r}
file_path <- "C:/Users/..."
```

```{r}
metadata <- read_tsv('metadata.tsv')
```

Are there missing values, and if so, in which columns?
```{r}
sum(is.na(metadata))
colnames(metadata)[colSums(is.na(metadata))>0]
```
We have 749 missing values in the next columns: 'consent_age','Age at diagnosis', 'fecalcal', 'BMI_at_baseline', Height_at_baseline', 'Weight_at_baseline' and 'smoking status'.

Amount of missing values for each feature, each of which is displayed in a column:
```{r}
col_na_count <- colSums(is.na(metadata))
barplot(col_na_count, main = "Amount of missing values for each feature", col = "lightblue", names.arg = colnames(metadata), las = 2, cex.names = 0.5)
```

The fecal calprotectin data contains NA values, they'll be removed.
```{r}
clean_data <- metadata %>% filter(!is.na(fecalcal)) 
print(clean_data) 
```
Checking if all the NA values are removed.
```{r}
sum(is.na(clean_data$fecalcal)) 
```
It is important to examine how the NA values are distributed across the three study groups. This helps understanding whether the missing data is randomly distributed or if there are systematic patterns. For example, if the NA values are concentrated in specific study groups, it may indicate issues in data collection or bias, which could impact the validity of further analyses.
```{r} 
na_distribution <- metadata %>% 
group_by(Study.Group) %>%  
summarise( 
  Total_NA = sum(is.na(fecalcal)), 
  Percentage_NA = (Total_NA / n()) * 100
) 
print(na_distribution) 
``` 
The distribution of missing values was examined, and percentage-wise, there is no notable difference between the three groups.

Visualising the distribution:
```{r} 
ggplot(na_distribution, aes(x = Study.Group, y = Percentage_NA, fill = Study.Group)) + geom_bar(stat = "identity") +
labs( 
  title = "Percentage of Missing Values per Study Group",
  x = "Study Group", 
  y = "Percentage of NA" 
) + 
theme_minimal() 
```
Percentage-wise, no notable differences in the distribution of missing values (NAs) are observed across the study groups. To determine if this is statistically significant, a chi-square test will be applied. This method is suitable for categorical variables and assesses associations under the assumption of random group distributions.

The aim is to provide evidence in favor of the null hypothesis (H₀), with statistical significance evaluated at the threshold of p > 0.05.
```{r} 
chisq_test_data <- metadata %>% 
  mutate(fecalcal_missing = ifelse(is.na(fecalcal), "Missing", "Not Missing")) %>% 
  count(Study.Group, fecalcal_missing) %>% 
  pivot_wider(names_from = fecalcal_missing, values_from = n, values_fill = 0) 

chisq_result <- chisq.test(chisq_test_data[,-1])  #Disregard the Study.Group column
print(chisq_result) 
```

#### Interpretation: 
Based on the p-value (0.8778), we fail to reject the null hypothesis (H₀), indicating that there is no significant difference in the number of missing values (NAs) across the study groups (CD, UC, non-IBD).

The missing values appear to be randomly distributed among the groups, suggesting that the missing data is likely missing completely at random (MCAR). This implies that the distribution of NAs does not show any systematic pattern, and therefore, it is unlikely to introduce bias into the analysis.

Now that the NA distribution has been checked, it is also important to verify whether there are any duplicate subjects, as these could potentially influence the results.
```{r} 
duplicated_subjects <- clean_data[duplicated(clean_data$Subject), ]
```

Retrieving the duplicate subjects:
```{r} 
cat("Aantal dubbele subjects:", nrow(duplicated_subjects), "\n")
print(duplicated_subjects) 
```

Duplicate subjects are removed by calculating the mean of the numerical fecalcal data for these subjects, and are is stored under a new feature called 'fecalcal_mean'.
```{r} 
metadata_clean <- clean_data %>% 
  group_by(Subject) %>% 
  filter(n_distinct(Study.Group) == 1) %>% 
  summarise( 
    Study.Group = first(Study.Group), 
    fecalcal_mean = mean(fecalcal, na.rm = TRUE), 
    .groups = "drop" 
  )
print(metadata_clean) 
```

Controlling if all the duplicate subjects have been removed:
```{r}
duplicated_subjects1 <- metadata_clean[duplicated(metadata_clean$Subject), ] 
cat("dubble subjects:", nrow(duplicated_subjects1), "\n") 
print(duplicated_subjects1) 
```

### Descriptive statistics:
Calculating the mean, median, SD, min, max of fecalcal_mean per Study.Group.
```{r} 
desc_stats <- metadata_clean %>% 
  group_by(Study.Group) %>% 
  summarize( 
    Mean = mean(fecalcal_mean), 
    Median = median(fecalcal_mean), 
    SD = sd(fecalcal_mean), 
    Min = min(fecalcal_mean), 
    Max = max(fecalcal_mean) 
  ) 
print(desc_stats) 
```

#### Interpretation:
We observe that the difference in the average values between CD and UC is relatively small. However, there is a much larger difference in the averages between both CD and non-IBD, as well as UC and non-IBD. Additionally, the minimum values across the study groups are very similar. In contrast, the maximum values are more spread out, except for between UC and CD. Notably, the maximum value for non-IBD is close to the median values of both UC and CD

Visualisation of the distribution:
```{r} 
ggplot(metadata_clean, aes(x = Study.Group, y = fecalcal_mean, fill = Study.Group)) + 
  geom_boxplot() + 
  geom_jitter(width = 0.2, alpha = 0.5) + 
  labs(title = "Calprotectine per Study group", y = "Calprotectine (µg/g)", x =      "Group") + 
  theme_minimal() 
``` 

Histogram of fecal calprotectine per group:
```{r} 
ggplot(metadata_clean, aes(x = fecalcal_mean, fill = Study.Group)) + 
  geom_histogram(binwidth = 50, alpha = 0.6, position = "identity") + 
  facet_wrap(~Study.Group) +  # Create separate plots for each group
  labs(title = "Histogram per Group", x = "Calprotectine (µg/g)", y = "Frequency") + 
  theme_minimal() 
``` 
From the boxplots and histogram, we observed that there are outliers in the non-IBD group. These can be filtered using IQR boundaries and subsequently excluded from the data. The outliers are removed to improve the statistical power of the test.

Calculating the IQR boundaries per group:
```{r} 
quartiles <- metadata_clean %>% 
  group_by(Study.Group) %>% 
  summarize( 
    Q1 = quantile(fecalcal_mean, 0.25), 
    Q3 = quantile(fecalcal_mean, 0.75) 
  ) %>% 
  mutate( 
    IQR = Q3 - Q1, 
    Lower_Bound = Q1 - 1.5 * IQR, 
    Upper_Bound = Q3 + 1.5 * IQR 
  ) 
```
Add bounderaries to the data:
```{r} 
clean_data_unique <- metadata_clean %>% 
  left_join(quartiles, by = "Study.Group") %>% 
  filter(fecalcal_mean>= Lower_Bound & fecalcal_mean <= Upper_Bound) 
```

Boxplot of fecal calprotectine per group WITHOUT outliers:
```{r} 
ggplot(clean_data_unique, aes(x = Study.Group, y = fecalcal_mean, fill = Study.Group)) + 
  geom_boxplot() + 
  geom_jitter(width = 0.2, alpha = 0.5) + 
  labs(title = "Fecalcal per Group (without outliers)", y = "Fecalcal (µg/g)", x = "Group") + 
  theme_minimal()
``` 

Histogram of fecal calprotectine per group WITHOUT outliers:
```{r} 
ggplot(clean_data_unique, aes(x = fecalcal_mean, fill = Study.Group)) + 
  geom_histogram(binwidth = 50, alpha = 0.6, position = "identity") + 
  facet_wrap(~Study.Group) +  
  labs(title = "Histogram per Group (without outliers)", x = "Calprotectine (µg/g)", y = "Frequency") + 
  theme_minimal()
```

#### Interpretation:
We have added 5 new variables to the clean_data_unique dataset:
- Q1: The first quartile (25th percentile).
- Q3: The third quartile (75th percentile).
- IQR: The difference between Q3 and Q1.
- Lower Bound = Q1 − 1.5 ⋅ IQR
- Upper Bound = Q3 + 1.5 ⋅ IQR

Outliers were only present in the non-IBD group. After removing the outliers, we are left with 78 subjects. In the histogram per group (without outliers) of clean_data_unique, we observe that the data do not follow a normal distribution, as no bell-shaped curve is seen in any of the study groups. Therefore, we will proceed with a non-parametric test, the Kruskal-Wallis test, to analyze the differences between the groups.

Kruskal-Wallis test on the clean_data_unique (without outliers and duplicate subjects).
```{r} 
install.packages("coin") 
library(coin) 
kruskal.test(fecalcal_mean ~ Study.Group, data = clean_data_unique) 
``` 

#### Interpretation:
There is a significant difference in fecal calprotectin levels across the three study groups (P = 0.0001005 << 0.05). To further differentiate between the groups, we will need to perform a post hoc analysis. 

Post-hoc analysis will be conducted using the pairwise Wilcoxon test to examine the differences between the groups.
```{r} 
pairwise_wilcox <- pairwise.wilcox.test( 
  x = clean_data_unique$fecalcal_mean, 
  g = clean_data_unique$Study.Group, 
  p.adjust.method = "bonferroni"  # Correction for multiple testing 
) 
print(pairwise_wilcox) 
``` 
#### Interpretation:
The output shows p-values from pairwise comparisons with a Bonferroni correction. Both CD and UC differ significantly from non-IBD in fecal calprotectin levels (p = 0.00014 and p = 0.00027). However, no significant difference exists between CD and UC (p = 1.0000), indicating that fecal calprotectin cannot distinguish between these two groups. It is effective in differentiating non-IBD from both UC and CD, but not between CD and UC.

Creating a boxplot using ggboxplot, with Wilcoxon p-values added to show statistical significance.
```{r}
install.packages("ggpubr")
library(ggpubr)
```

```{r} 
ggboxplot( 
  clean_data_unique,  
  x = "Study.Group",  
  y = "fecalcal_mean",  
  fill = "Study.Group",  
  add = "jitter",  # Add points to show data distribution 
  palette = "jco"  # Choose a color scheme 
) + 
  stat_compare_means( 
    method = "kruskal.test",  # Kruskal-Wallis for global comparison 
    label.y = max(clean_data_unique$fecalcal_mean) * 1.1  # Position global p-value 
  ) + 
  stat_compare_means( 
    comparisons = list(c("CD", "UC"), c("CD", "non-IBD"), c("UC", "non-IBD")),  # Comparisons 
    method = "wilcox.test",  
    p.adjust.method = "bonferroni",  # Correction for multiple testing 
    label = "p.signif"  # Use *, **, *** for p-values 
  ) + 
  labs( 
    title = "Calprotectin Levels per Study Group", 
    x = "Group", 
    y = "Calprotectin (µg/g)" 
  )
``` 

#### Interpretation: 
Since UC and CD do not show a significant difference in fecal calprotectine levels, they can be grouped together as IBD (Inflammatory Bowel Disease) and compared to non-IBD. This allows for a broader comparison between IBD and non-IBD based on fecalcal levels."

```{r} 
install.packages("dplyr") 
library(dplyr) 
```

Creating a new column where UC and CD are combined into IBD:
```{r} 
clean_data_unique <- clean_data_unique %>% 
  mutate(Group_Combined = ifelse(Study.Group %in% c("UC", "CD"), "IBD", Study.Group)) 
``` 

Kruskal-Wallis Test for IBD vs Non-IBD:

```{r} 
kruskal_combined <- kruskal.test(fecalcal_mean ~ Group_Combined, data = clean_data_unique) 
print(kruskal_combined) 
```

#### Interpretation:
The comparison between the non-IBD and IBD groups is statistically significant (p = 1.79e-5 < 0.05), suggesting that fecal calprotectin levels can effectively distinguish between these groups.

For the confidence intervals, it is preferable to use the median since our data is not normally distributed. The median is much more robust to outliers than the mean.

```{r} 
install.packages("Hmisc") 
library(Hmisc)
set.seed(123)  # For reproducibility
library(dplyr)
```

Perform bootstrapping to calculate the confidence intervals for the median of each group:
```{r} 
bootstrap_ci <- function(data, group, value, n_boot = 1000, conf_level = 0.95) { 
  groups <- unique(data[[group]]) 
  results <- data.frame(Group = character(0), CI_Lower = numeric(0), CI_Upper = numeric(0)) 
  
  for (g in groups) { 
    group_data <- data %>% filter(!!sym(group) == g) 
    medians <- replicate(n_boot, median(sample(group_data[[value]], replace = TRUE))) 
    lower <- quantile(medians, (1 - conf_level) / 2) 
    upper <- quantile(medians, 1 - (1 - conf_level) / 2) 
    results <- rbind(results, data.frame(Group = g, CI_Lower = lower, CI_Upper = upper)) 
  } 
  
  return(results) 
}
```

Calculate the 95% confidence intervals for the median of the fecalcal_mean variable for each group:
```{r} 
ci_results <- bootstrap_ci( 
  data = clean_data_unique, 
  group = "Group_Combined", 
  value = "fecalcal_mean", 
  n_boot = 1000, 
  conf_level = 0.95 
)

print(ci_results)
```

#### Interpretation:
The 95% confidence intervals (CIs) for median fecal calprotectin levels show distinct patterns between the IBD and non-IBD groups. The IBD group has a CI from 41.48 µg/g to 147.69 µg/g, indicating elevated median levels with some variability. In contrast, the non-IBD group has a narrower CI (14.87 µg/g to 22.72 µg/g), suggesting lower and more consistent levels. The non-overlapping CIs between these groups indicate a statistically significant difference, supporting fecal calprotectin as a reliable biomarker to distinguish IBD from non-IBD.

The Kruskal-Wallis test is used to determine if there is a significant difference in the median fecal calprotectin levels between the "IBD" and "non-IBD" groups:
```{r} 
kruskal_combined <- kruskal.test(fecalcal_mean ~ Group_Combined, data = clean_data_unique) 
```

Calculating the effect size (eta-squared, η²):
```{r}
H <- kruskal_combined$statistic
k <- length(unique(clean_data_unique$Group_Combined))
n <- nrow(clean_data_unique)

# Calculate eta-squared
eta_squared <- (H - k + 1) / (n - k)
print(eta_squared)
```

#### Interpretation:
The output provides the effect size (η²) of the Kruskal-Wallis test. A value of 0.229 (rounded) indicates that approximately 22.9% of the variance in calprotectin levels can be explained by the differences between the groups (IBD vs non-IBD). This suggests that the group classification has a notable impact on the variation observed in calprotectin levels.

### **Conclusion**:
Significant differences in fecal calprotectin levels were found across the three study groups (P = 0.0001005 << 0.05). Pairwise comparisons with a Bonferroni correction revealed that both CD and UC differ significantly from non-IBD (p = 0.00014 < 0.05 and p = 0.00027 < 0.05), but no significant difference was observed between CD and UC (p = 1.0000 > 0.05). This allows UC and CD to be grouped together as IBD, distinguishing them from non-IBD, with a statistically significant difference between IBD and non-IBD (p = 1.79e-5 < 0.05).

The 95% confidence intervals (CIs) further support this conclusion, showing non-overlapping CIs that clearly differentiate IBD from non-IBD. The IBD group has a CI ranging from 41.48 µg/g to 147.69 µg/g, while the non-IBD group’s CI is narrower, ranging from 14.87 µg/g to 22.72 µg/g. Additionally, the effect size (η² = 0.229) from the Kruskal-Wallis test indicates that 22.9% of the variance in calprotectin levels can be explained by group differences, further confirming the ability of fecal calprotectin to differentiate between these conditions.

Thus, fecal calprotectin levels can be used as a reliable diagnostic biomarker to differentiate between non-IBD and ulcerative colitis (UC) or Crohn's disease (CD), but not between UC and CD. These findings underscore fecal calprotectin as a reliable biomarker for distinguishing IBD (UC & CD) from non-IBD.

---------------------------------------------------------------------------------------

# Research question 2 (multivariate analysis): 

## Are bacterial classes associated with Non-IBD, Ulcerative Colitis (UC), and Crohn's Disease (CD) in terms of microbiome composition?

**Null hypothesis:** There is no significant difference in bacterial composition between the three study groups (class CD, UC and non-IBD).

**Alternative hypothesis:** There is significant difference in bacterial composition between the three study groups (class CD, UC and non-IBD).

### Data preparation:

Needed libraries
```{r}
# Install the package by removing #, if not already installed
# install.packages("tidyverse")
# install.packages("ggplot2")
# install.packages("vegan")
# install.packages("ggcorrplot")
# install.packages("dplyr")

library(tidyverse)
library(ggplot2)
library(vegan) # For adonis2 function
library(ggcorrplot)
library(dplyr)
```

```{r}
file_path <- "C:/Users/..."
```

Read data
```{r}
genera_counts <- read_tsv('genera.counts.tsv')
metadata <- read_tsv('metadata.tsv')
```

Check for missing values.
```{r}
sum(is.na(genera_counts))
```
We don't have missing data values. Now we need to know what the different bacterial classes are.

```{r}
column_names <- colnames(genera_counts)

extract_class_from_column_name <- function(column_name) {
  match <- regmatches(column_name, regexec("c__([A-Za-z0-9_-]+)", column_name))
  
  if (length(match[[1]]) > 1) {
    return(match[[1]][2])  
  }
  return(NA)  # No match found
}

# Obtain a vector of classes for each column
classes <- sapply(column_names, extract_class_from_column_name)

unique_classes <- unique(classes)

cat("Unique classes in the dataset:\n")
print(unique_classes)
```
We observe a total of 275 unique classes, excluding the first generated output, which is treated as NA because it originates from the Sample column.


Now we are going to replace all bacterial column names with the bacterial class names:
 
```{r}
# Function to rename column names
rename_columns <- function(column_names, classes) {
  new_column_names <- character(length(column_names))
  
  # Loop over the column names
  for (i in seq_along(column_names)) {
    class_name <- classes[i]
    
    # Only continue if the class_name is valid (not NA)
    if (!is.na(class_name)) {
      new_column_names[i] <- class_name  # No index added, just class name
    } else {
      new_column_names[i] <- "Unknown"  # If no class is found, call the column "Unknown"
    }
  }
  
  return(new_column_names)
}

# Apply function to rename column names
new_column_names <- rename_columns(column_names, classes)

# Rename the columns in the dataset
colnames(genera_counts) <- new_column_names

# Check out the updated column names
print(colnames(genera_counts))
head(genera_counts)
```	


Samples column is currently labeled as 'Unknown'. Check if any other column is labeled as 'Unknown' as well.


```{r}
# Count how many times ‘Unknown’ appears in the column names
unknown_count <- sum(colnames(genera_counts) == "Unknown")

# Print the result
cat("Aantal 'Unknown' kolommen:", unknown_count, "\n")
```

Now, merge the columns with the same name. Only the column 'Sample' was named 'Unknown'. So, rename the 'Unknown' column back to 'Sample'.
```{r}
# Step 1: Keep ‘Unknown’ untouched and rename to ‘Sample’
genera_counts$Sample <- genera_counts$Unknown
genera_counts <- genera_counts[, names(genera_counts) != "Unknown"]

# Step 2: Check and convert only columns that are numeric
genera_counts_numeric <- genera_counts
genera_counts_numeric[] <- lapply(genera_counts_numeric, function(col) {
  if (is.numeric(col)) {
    col # If the column is already numeric, leave it untouched
  } else if (is.factor(col) || is.character(col)) {
    suppressWarnings(as.numeric(col)) # Try converting text/factor to numeric
  } else {
    NULL # Exclude columns that cannot be converted
  }
})

# Step 3: Combine columns with the same name by merging them together
genera_counts_combined <- as.data.frame(sapply(unique(names(genera_counts_numeric)), function(col) {
  if (col %in% names(genera_counts_numeric)) {
    rowSums(genera_counts_numeric[names(genera_counts_numeric) == col], na.rm = TRUE)
  } else {
    NULL
  }
}))

# Step 4: Add the ‘Sample’ column back to the merged dataset
genera_counts_combined$Sample <- genera_counts$Sample

# Step 5: Move the ‘Sample’ column to the first position
genera_counts_combined <- genera_counts_combined[, c("Sample", setdiff(names(genera_counts_combined), "Sample"))]

# Step 6: Check the result
print(genera_counts_combined)
```
Append the 'Study.Group' column from the dataset 'metadata' to the 'genera_counts_combined' dataset.

```{r}
genera_counts_combined <- merge(genera_counts_combined, metadata[, c("Sample", "Study.Group")], by = "Sample")
```
Append the 'Subject' column from the dataset 'metadata' to the 'genera_counts_combined' dataset.

```{r}
genera_counts_combined <- merge(genera_counts_combined, metadata[, c("Sample", "Subject")], by = "Sample")
```

Check the number of duplicate subjects.
```{r}
# Check for duplicate values in the ‘Subject’ column
duplicated_subjects <- genera_counts_combined[duplicated(genera_counts_combined$Subject), ]

# Print the number of duplicate values
cat("Amount of duplicate subjects:", nrow(duplicated_subjects), "\n")

# View duplicate values
print(duplicated_subjects)
```
The dataset contains 277 duplicate subjects.

Now take the average of the double subjects.
```{r}
# Calculate the average for Subjects and retain Study.Group
genera_counts_combined_clean <- genera_counts_combined %>%
  group_by(Subject) %>%
  summarise(
    Study.Group = first(Study.Group), 
    across(where(is.numeric), mean, na.rm = TRUE),
    .groups = "drop"
  )

# View the resulting dataset
print(genera_counts_combined_clean)

# Check the number of unique subjects
cat("Amount of unique subjects:", n_distinct(genera_counts_combined_clean$Subject), "\n")
```
We now have 105 unique subjects.


### Correlation:

Before performing a correlation matrix analysis, it's important to handle values that may distort the correlation calculations. Specifically, zeros in the dataset might represent missing data or irrelevant values, but they can skew the correlation results if treated as valid numerical values. Since correlation calculations cannot work correctly with zeros representing missing data, we can replace all zeros with NA (Not Available), which will exclude them from the correlation matrix.

This ensures that zeros are not mistakenly treated as valid values during the correlation calculation, and the correlation matrix will be computed based only on the non-zero (and meaningful) data.


```{r}
# Replace all 0 values with NaN in numeric columns
genera_counts_combined_clean_withNan <- genera_counts_combined_clean %>%
  mutate(across(where(is.numeric), ~ replace(., . == 0, NaN)))

# View the modified dataset
print(genera_counts_combined_clean_withNan)

# Check the number of NaN values in the dataset
cat("Number of NaN values in the dataset:", sum(is.nan(as.matrix(genera_counts_combined_clean_withNan))), "\n")
```

Visualize the correlation matrix:
```{r}
# Select only numeric variables
numeric_vars <- sapply(genera_counts_combined_clean_withNan, is.numeric)
correlation_matrix <- cor(genera_counts_combined_clean_withNan[, numeric_vars], use = "pairwise.complete.obs")

# Visualising the correlation matrix
ggcorrplot(correlation_matrix, 
           lab = FALSE, 
           title = "Correlation Matrix", 
           outline.col = "white")
```
The correlation matrix reveals that the dataset contains a large number of variables, indicating high dimensionality. It shows a spread of correlations: some variables exhibit strong positive or negative correlations, while others are scarcely correlated.

Therefore, applying a technique such as Principal Component Analysis (PCA) is valuable. PCA can help reduce dimensionality by identifying the most significant components in the data, while retaining a large portion of the variation. This simplifies and makes the dataset more efficient for further analysis.

### Dimensionality reduction:

```{r}
# Step 1: Perform the PCA on the numeric columns
bacteria_data <- genera_counts_combined_clean %>%
  select(-c(Subject, Study.Group))  # Remove the non-numeric columns

# Run PCA
pca_result <- prcomp(bacteria_data)

# Step 2: Extract the first two components for visualisation
pca_data <- as.data.frame(pca_result$x)
pca_data$Study.Group <- genera_counts_combined_clean$Study.Group

# Step 3: Visualise PCA with ggplot2
library(ggplot2)
ggplot(pca_data, aes(x = PC1, y = PC2, color = Study.Group)) +
  geom_point(size = 3) +
  labs(title = "PCA of bacterial distribution",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

```
The code aims to reduce the dimensionality of the bacterial abundance data using PCA, and then visualize the data in two dimensions (PC1 and PC2) to see how different study groups (UC, CD, non-IBD) cluster or separate from each other based on their bacterial profiles.

Check for possible outliers:
```{r}
# Calculate Mahalanobis distance in head space
pca_scores <- pca_result$x[, 1:2]  # The first two main components
mahalanobis_pca <- mahalanobis(pca_scores, colMeans(pca_scores), cov(pca_scores))

# Add the distance to your data frame
genera_counts_combined_clean$mahalanobis_pca <- mahalanobis_pca

# Threshold for identifying outliers
threshold_pca <- qchisq(0.95, df = 2)  # 95% threshold, 2 dimensions
outliers_pca <- which(mahalanobis_pca > threshold_pca)

# Visualise the outliers
plot(pca_scores, col = ifelse(mahalanobis_pca > threshold_pca, "red", "black"))
```
We have decided to retain the outliers. Retaining outliers in PCA allows for capturing important, rare variations and reflecting the true complexity and diversity of real-world data, ensuring that meaningful patterns and extreme but legitimate variations are not lost.

We used a screeplot to identify the optimal number of principal components to retain by showing the variance explained by each component and highlighting the point where additional components contribute less.
```{r}
screeplot (pca_result, type='lines',main="PC Variance by PC # (Screeplot) ")
abline (h=mean ( (pca_result$sdev)^2), col= 'gray' , lty=2)
legend ("right", "Mean Variance" ,lty=2, col='gray',bty='n')
cumulative_variance <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
num_pcs <- which(cumulative_variance >= 0.9)[1]  # First PC to explain ≥90%
print(num_pcs)
```
Look for the "elbow point," where the additional variance explained by subsequent principal components becomes minimal, and select the components that together explain more than 90% of the total variance, here PC2.

```{r}
# Extract loadings for the principal components
pca_loadings <- as.data.frame(pca_result$rotation)
most_influential_classes <- rownames(pca_loadings[order(abs(pca_loadings$PC2), decreasing = TRUE)[1:5], ])
```
From this, we can identify which classes play the most important role and use them to examine whether these differ across study groups. Taking the highest variables (or loadings) of a principal component, means identifying the original variables that contribute most significantly to that component. These variables have the strongest influence on the direction and variance explained by the component, highlighting the key factors or features driving the patterns captured by that principal component. Here we used principal component 2. 

```{r}
# PCA loadings as a data frame
pca_loadings <- as.data.frame(pca_result$rotation)

# Find the top 5 most influential features for each PC
top_influential_classes <- lapply(names(pca_loadings), function(pc) {
  # Extract the current PC's loadings
  current_pc <- pca_loadings[[pc]]
  
  # Find the indices of the top 5 absolute loadings
  top_indices <- order(abs(current_pc), decreasing = TRUE)[1:5]
  
  # Return the feature names and their loadings
  data.frame(Feature = rownames(pca_loadings)[top_indices],
             Loading = current_pc[top_indices],
             PC = pc)
})

# Combine results into a single data frame
top_influential_classes <- do.call(rbind, top_influential_classes)

# View the result
print(top_influential_classes)
```
The top 5 bacterial classes are; Clostridia, Bacteroidia, Gammaproteobacteria, Negativicutes, and Bacilli.

### Descriptive Statistics:

Calculating the mean, median, and standard deviation for the top 5 bacterial classes:
```{r}
# Include Study.Group in the bacteria_data subset
bacteria_data <- genera_counts_combined_clean[, c("Clostridia", "Bacteroidia", "Gammaproteobacteria", "Negativicutes", "Bacilli", "Study.Group")]

# Calculate mean, median, and standard deviation per bacterial class grouped by Study.Group
mean_median_sd <- bacteria_data %>%
  group_by(Study.Group) %>%
  summarise(across(
    where(is.numeric),
    list(mean = mean, median = median, sd = sd),
    na.rm = TRUE
  ))

# Print the results
print(mean_median_sd)
```
NonIBD consistently exhibits higher microbial levels, as indicated by the elevated means and medians for taxa such as Clostridia and Bacteroidia. This pattern may reflect healthier or more stable microbial communities in individuals without inflammatory conditions. In contrast, CD and UC display greater variability, particularly evident in the higher standard deviations for taxa like Gammaproteobacteria in CD. This increased variability could be indicative of dysbiosis commonly associated with inflammatory diseases.

### Normality Testing:
To determine if the five bacterial classes are normally distributed, we created histograms for each bacterial class by study group and conducted a Shapiro-Wilk test.
```{r}
# List of bacteria classes
bacterial_classes <- c("Clostridia", "Bacteroidia", "Gammaproteobacteria", "Negativicutes", "Bacilli")

# For each bacterial class, create histograms by Study.Group
for (class in bacterial_classes) {
  plot <- ggplot(genera_counts_combined_clean, aes_string(x = class, fill = "Study.Group")) +
    geom_histogram(bins = 30, alpha = 0.7, position = "dodge", color = "black") +
    facet_wrap(~Study.Group, scales = "free") +  # Put facets per Study.Group
    labs(
      title = paste("Histogram of", class, "per Study Group"),
      x = class,
      y = "Frequency"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))  # Title centred
  print(plot)
}

# Shapiro-Wilk test for each bacterial class
shapiro_results <- sapply(genera_counts_combined_clean[bacterial_classes], function(x) shapiro.test(x)$p.value)

# Display results
shapiro_results
```
If the p-value of the Shapiro-Wilk test is greater than 0.05, the data can be considered normally distributed; otherwise, a p-value below 0.05 indicates significant deviation from normality. In this case, only one value exceeds 0.05, suggesting that most of the data significantly deviate from a normal distribution.

### PERMANOVA:
```{r}
bacteria_data <- genera_counts_combined_clean[, c("Clostridia", "Bacteroidia", "Gammaproteobacteria", "Negativicutes", "Bacilli")]
study_group <- genera_counts_combined_clean$Study.Group

# Perform PERMANOVA
perm_result <- adonis2(bacteria_data ~ study_group , method = "euclidean", permutations = 999)
print(perm_result)
```
#### Interpretation PERMANOVA:
R2 is the proportion of total variation explained by the study group factor is 0.01432. This means that only about 1.43% of the total variation in bacterial composition can be explained by the study group classification.

The F-statistic is 0.7408, , which measures the ratio of between-group variation to within-group variation. A higher F-value suggests more separation between groups, but this value is relatively low here.

The p-value is 0.52, which is greater than 0.05, indicating that there is no significant difference in the bacterial compositions between the study groups based on the Euclidean distance. 

### Conclusion:
The PERMANOVA results indicate that the null hypothesis cannot be rejected. This suggests:
- No significant association exists between the bacterial composition and study groups (Non-IBD, UC, CD).
- The small R² value highlights that only a tiny fraction of variance in bacterial composition is attributable to the study group factor.



# Machine learning:
Our goal is to develop a machine learning model within the context of the study 'Multi-omics of the gut microbial exosystem in Inflammatory Bowel Diseases (IBD). The aim is to make a model that can predict the study group to which a patient belongs - UC (Ulcerative Colitis), CD (Crohn’s Disease), or nonIBD— based on relevant features extracted from the dataset (metadata). 

Methodology:
We used Python as the primary programming environment for this analysis. Two machine learning models were implemented and compared: Logistic Regression and Random Forest. Logistic Regression provides insights into the relationship between features and the target variable through interpretable coefficients, while Random Forest leverages an ensemble of decision trees to handle more complex patterns within the data. Model performance was evaluated using metrics such as accuracy, precision, recall, F1-score, and mean absolute error (MAE).

### DATA EXPLORATION 
```{python}
import pandas as pd
```
```{python}
metadata = pd.read_csv('/kaggle/input/metadata-tsv/metadata.tsv',sep='\t')
```
```{python}
metadata.head()
```
```{python}
metadata.Subject
```
#### Making sure every subject has only one sample in the dataset 
Problem: Some patients in this dataset have multiple samples. To ensure there is only one row per patient, we can combine the rows coming from 1 subject.

First, we need to identify which patients (=subjects) have multiple samples:
```{python}
# Number of unique Subject values
unique_subjects = metadata['Subject'].nunique()
print(f"Number of unique Subject values: {unique_subjects}")

# Frequency of each unique value
subject_counts = metadata['Subject'].value_counts()

# Subject values that occur multiple times
duplicates = subject_counts[subject_counts > 1]
print("Subjects that occur multiple times:")
print(duplicates)
```
Now we want each subject to occur only once (i.e., 1 row per subject!) Of the numeric columns, we therefore calculate the average between the samples coming from the same patient. For categorical columns, we take the first value that occurs for the patient:
```{python}
# Split numeric and non-numeric columns
numeric_cols = metadata.select_dtypes(include=['number']).columns
non_numeric_cols = metadata.select_dtypes(exclude=['number']).columns

# Group by 'Subject' and apply aggregations
aggregated_data = metadata.groupby('Subject').agg(
    {col: 'mean' for col in numeric_cols} | {col: 'first' for col in non_numeric_cols}
)

# Reset index only if 'Subject' is not already a column
if 'Subject' not in aggregated_data.columns:
    aggregated_data.reset_index(inplace=True)
```{python}
print(aggregated_data.head())
```
```{python}
aggregated_data.shape
```
Next we do a small test to see if combining the different samples from one subject has been successful. We test with the subject 'H4008', which has 2 samples present in the original dataset.
```{python}
# Filter all rows with Subject 'H4008' in the original dataset
original_rows = metadata[metadata['Subject'] == 'H4008']
print("Original dataset (metadata):")
print(original_rows)

# Filter the row with Subject 'H4008' in the aggregated dataset
aggregated_row = aggregated_data[aggregated_data['Subject'] == 'H4008']
print("\nAggregated dataset (aggregated_data):")
print(aggregated_row)
```
```{python}
# Checking the number of rows for Subject H4008
# Number of rows in the original dataset
print(f"Number of rows in metadata for 'H4008': {len(original_rows)}")

# Number of rows in the aggregated dataset
print(f"Number of rows in aggregated_data for 'H4008': {len(aggregated_row)}")
```
We see that in the aggregated dataset, the subject 'H4008' has only one row.
```{python}
# Now change the name to of the aggregated data to metadata_adjusted to avoid confusion:
metadata_adjusted = aggregated_data
```
```{python}
#looking at the shape of the dataset, how many rows and how many columns are there now?
metadata_adjusted.shape
```
```{python}
print(metadata_adjusted)
```
### FEATURES 
Firstly, we explored which features were interesting for predicting the studygroup (CD, UC, nonIBD). 

For our machine learning model, the interesting features are: 
* week_num
* interval_days
* Age at diagnosis
* fecalcal
* BMI_at_baseline
* Height_at_baseline
* Weight_at_baseline
* Study.Group
* Gender
* Antibiotics
* race
* smoking status 

Then we do a little bit of exploration on these features, do they have a lot of missing values?
```{python}
# Check for the number of NA values in each column
na_counts = metadata_adjusted.isna().sum()

# Display the number of missing values for each column
print(na_counts)
```
```{python}
# Display columns with NA values
na_counts = metadata_adjusted.isna().sum()
na_columns = na_counts[na_counts > 0]

# Print columns with missing values
print(na_columns)
```
Unfortunately, many of the key features contain a substantial amount of missing values ("NA"). If we choose to include these features in our predictive model, we would need to remove the rows with missing data. However, this would significantly reduce the number of usable rows, so it may be more effective to focus on the features that don't have missing values

### CONVERSION TO NUMERIC VALUES 
We have to convert the non-numeric features to numeric features because: 
*  Logistic regression requires that the data fed into it consists of numerical values, as it performs mathematical operations on the features, such as calculating weights, coefficients, and other numerical measures.
*  Random forest can also not work with non-numeric data.

We convert the non-numeric features that may be interesting for our machine learning model into numeric:
```{python}
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Variables to encode
categorical_columns = ['Study.Group','Gender', 'Antibiotics', 'race', 'smoking status']

# Loop through the columns and apply label encoding
for column in categorical_columns:
    metadata_adjusted[f'{column}_encoded'] = label_encoder.fit_transform(metadata_adjusted[column])
    print(f"Mapping for {column}:")
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print()  # Empty line for readability

# View the result (e.g., the first few rows)
print(metadata_adjusted[[col for col in categorical_columns] + [f'{col}_encoded' for col in categorical_columns]].head())
```
From now on, we need to use the new '-encoded' names for the columns:

Study.Group -> 'Study.Group_encoded'
Gender -> 'Gender_encoded'
Anitbiotics -> 'Antibiotics_encoded'
race -> 'race_encoded'
smoking status -> 'smoking status_encoded'

## MACHINE LEARNING MODEL 1 
### 1) Defining target value y and features X 
For our first machine learning model:
* our y = Study.Group_encoded
* our IBD_features (X) = interval_days, visit_num, Gender_encoded, Anitbiotics_encoded, Race_encoded, BMI_at_baseline, Height_at_baseline, Weight_at_baseline

Due to the missing values that are present in the features 'BMI_at_baseline, Height_at_baseline, Weight_at_baseline' we first need to remove the rows with missing values:
```{python}
#removing missing values in the features BMI_at_baseline, Height_at_baseline, Weight_at_baseline
metadata_adjusted_withoutNA = metadata_adjusted.dropna(subset=['BMI_at_baseline',
       'Height_at_baseline', 'Weight_at_baseline'], axis=0)
```
```{python}
#preparing the data for training a machine learning model by setting up the features (X) and the target variable (y)
y = metadata_adjusted_withoutNA['Study.Group_encoded']
IBD_features = ['interval_days',
       'visit_num', 'Gender_encoded',
       'Antibiotics_encoded', 'race_encoded','BMI_at_baseline',
       'Height_at_baseline', 'Weight_at_baseline']
X = metadata_adjusted_withoutNA[IBD_features]
```
```{python}
#get dimensions (number of rows, number of columns) of y 
y.shape
```
```{python}
#get dimensions (number of rows, number of columns) of X
X.shape
```
With this we can see that after removing the missing values, we are left with 81 remaining rows. And we can see that we have 8 features that are represented by the columns

### 2) Splitting data into train and test sets 
Then we split our dataset into training and test sets for building our machine learning model:
```{python}
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
```
### 3) PREPROCESSING DATA: Normalise feature values 
Next, we take a look at the distribution of our feature values, both for the train set and the test set
```{python}
import pandas as pd
import matplotlib.pyplot as plt

import warnings;
warnings.filterwarnings('ignore');
```
```{python}
#plotting the distrubution of the train set 
plt.figure(figsize=(18,6)) #set the size of the plot
train_X.sample(8, axis="columns").boxplot()
plt.xticks(rotation=90) #rotate the gene names on the x-axis
plt.show() #show the plot
```
```{python}
#plotting the distribution of the test set 
plt.figure(figsize=(18,6)) #set the size of the plot
val_X.sample(8, axis="columns").boxplot()
plt.xticks(rotation=90) #rotate the gene names on the x-axis
plt.show() #show the plot
```
The features clearly have very different scales. It is good practice to normalize the range of the features in the dataset.

We can do this with the Scikit-learn object StandardScaler:
```{python}
from sklearn.preprocessing import StandardScaler

#First we create a StandardScaler object called scaler_std: 
scaler_std = StandardScaler()
```
```{python}
#We first compute the mean and standard deviation from the train set:
scaler_std.fit(train_X)

print("Means:")
print(scaler_std.mean_)
print("Variances:")
print(scaler_std.var_)
```
```{python}
#We then compute the mean and standard deviation from the test set:
scaler_std.fit(val_X)

print("Means:")
print(scaler_std.mean_)
print("Variances:")
print(scaler_std.var_)
```
Next, we normalize the features in the train and test set.

The scaler object has a function transform() that uses the means and variances computed by the fit() function to normalize the features and we can make a Pandas DataFrame with the original column names as follows:
```{python}
#normalising the features if the train set 
train_X_std = scaler_std.transform(train_X)
train_X_std = pd.DataFrame(train_X_std, columns=train_X.columns)

train_X_std
```
```{python}
# normalising the features of the test set 
val_X_std = scaler_std.transform(val_X)
val_X_std = pd.DataFrame(val_X_std, columns=train_X.columns)

val_X_std
```
We can now plot the normalized features:
```{python}
#plotting distribution of the features of the train set 
plt.figure(figsize=(18,6))
train_X_std.sample(8, axis="columns").boxplot()
plt.xticks(rotation=90)
plt.show()
```
```{python}
#plotting distribution of the features of the train set
plt.figure(figsize=(18,6))
val_X_std.sample(8, axis="columns").boxplot()
plt.xticks(rotation=90)
plt.show()
```
As we can see, the feature value distributions are now normalized for both the train set and the test set

### 4) FIT LOGISTIC REGRESSION MODEL 
Now we can fit a logistic regression model.
In Scikit-learn we first initiate a LogisticRegression mode ! 
Train the model on the normalized train set:
```{python}
from sklearn.linear_model import LogisticRegression

cls_std = LogisticRegression()
```

```{python}
# Import the LogisticRegression model
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
cls_std = LogisticRegression()

# Fit the model on the normalized train set (features train_X_std and target train_y)
cls_std.fit(train_X_std, train_y)

# Optionally, print the model's parameters to check if it trained successfully
print("Model coefficients: ", cls_std.coef_)
print("Model intercept: ", cls_std.intercept_)
```

```{python}
#visualise model coefficients and model interecpt in plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Feature names and model parameters
feature_names = train_X_std.columns  # Assuming train_X_std is a DataFrame
coefficients = cls_std.coef_
intercepts = cls_std.intercept_

# Convert coefficients to a DataFrame for better visualization
coeff_df = pd.DataFrame(coefficients, columns=feature_names)
coeff_df['Class'] = [f"Class {i}" for i in range(coefficients.shape[0])]

# Melt the DataFrame for easier plotting
coeff_melted = coeff_df.melt(id_vars='Class', var_name='Feature', value_name='Coefficient')

# Plot coefficients
plt.figure(figsize=(12, 8))
sns.barplot(data=coeff_melted, x='Feature', y='Coefficient', hue='Class')
plt.title("Logistic Regression Coefficients per Class")
plt.xticks(rotation=45, ha="right")
plt.axhline(0, color='grey', linestyle='--')  # Line at 0 for reference
plt.tight_layout()
plt.show()

# Display intercepts separately
plt.figure(figsize=(6, 4))
sns.barplot(x=[f"Class {i}" for i in range(len(intercepts))], y=intercepts, palette="muted")
plt.title("Logistic Regression Intercepts per Class")
plt.ylabel("Intercept Value")
plt.axhline(0, color='grey', linestyle='--')
plt.tight_layout()
plt.show()
```
#### INTERPRETATION MODEL PARAMETERS
The model parameters represent a multiclass logistic regression with three classes:

Model Coefficients:
* Each row corresponds to one class, and each column represents a feature.
* Positive coefficients: Features that increase the probability of predicting a given class as the feature value increases. -> Example: "Gender_encoded" positively influences Class 0.
* Negative coefficients: Features that decrease the probability of predicting a given class as the feature value increases. -> Example: "Gender_encoded" has a strong negative influence for Class 1.
* For example, the feature with a positive coefficient (e.g., 0.291 for class 1) increases the odds of predicting class 1, while a negative coefficient (e.g., -0.547 for class 2) reduces it.
* Larger bars (either positive or negative) indicate features that have a stronger influence on the prediction for that class. Smaller bars (near zero) indicate that the feature has little impact on the model's prediction for that class. -> "Height_at_baseline" has a large positive coefficient for Class 1, meaning it strongly increases the likelihood of predicting Class 1. -> "BMI_at_baseline" has a large positive coefficient for Class 2.
Model Intercepts:
* The intercepts represent the baseline log-odds for each class when all features are zero.
* The intercept for class 1 (0.789) means this class has a higher likelihood of being predicted by default compared to classes 2 and 3 with lower intercepts (-0.521, -0.268).

The model appears to have learned meaningful relationships, as indicated by the variation in coefficients. However, some features dominate the predictions, which may increase the risk of overfitting or reliance on a small set of features.

Computing the accuracy: 
```{python}
# Compute predictions on the validation set val_X_std
predictions_std = cls_std.predict(val_X_std)

# Optionally, print the predicted class labels for each row in val_X_std
print(predictions_std)

# Alternatively, to get the class probabilities instead of predicted classes:
predictions_std_proba = cls_std.predict_proba(val_X_std)

# Display the first 10 predicted class probabilities (for both class 0 and class 1)
print(predictions_std_proba[:10])

# Compute the accuracy of the model's predictions on the validation set
accuracy = cls_std.score(val_X_std, val_y)
print("Accuracy: {}".format(accuracy))
```
#### INTERPRETATION ACCURACY
* The accuracy of this machine learning model is 42.86% (approximately 0.43), meaning the model correctly classified about 43% of the data points
* this low accuracy suggests that the model performs only slightly better than random guessing for a multi-class classification problem
* this also suggests that the model is struggling to generalize well, particularly in distinguishing between classes.

We calculate some other metrics to evaluate the model training
```{python}
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

precision = precision_score(val_y, predictions_std, average='weighted')
recall = recall_score(val_y, predictions_std, average='weighted')
f1 = f1_score(val_y, predictions_std, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Confusion Matrix
cm = confusion_matrix(val_y, predictions_std)
print("Confusion Matrix:\n", cm)
```
#### Visualisation other metrics
```{python}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Example predictions (replace these with your actual data)
predictions_std = cls_std.predict(val_X_std)
accuracy = cls_std.score(val_X_std, val_y)
precision = precision_score(val_y, predictions_std, average='weighted')
recall = recall_score(val_y, predictions_std, average='weighted')
f1 = f1_score(val_y, predictions_std, average='weighted')
cm = confusion_matrix(val_y, predictions_std)

# Create a dataframe for metrics
data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
}
metrics_df = pd.DataFrame(data)

# Plot the table and confusion matrix
fig, ax = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 2]})

# Add the main title
fig.suptitle('Model 1', fontsize=14, fontweight='bold')

# Table for metrics
ax[0].axis('tight')
ax[0].axis('off')
table = ax[0].table(cellText=metrics_df.values,
                    colLabels=metrics_df.columns,
                    loc='center',
                    cellLoc='center', rowLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
# Confusion matrix heatmap
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues', ax=ax[1], colorbar=False)
ax[1].set_title('Confusion Matrix', fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the suptitle
plt.show()
```

#### INTERPRETATION OTHER METRICS
* Precision = 0.561 => When the model predicts a class as positive, it is correct 56.1% of the time. This indicates moderate precision.
* Recall = 0.429 => The model correctly identifies 42.9% of the actual positive cases, showing it misses many true positives.
* F1-Score = 0.330 => The harmonic mean of precision and recall is low (33%), indicating overall poor balance between precision and recall.
The confusion matrix shows:
* Class 0 is predicted reasonably well (8 correct, 1 misclassified).
* Class 1 is poorly predicted (7 misclassified as Class 0, only 1 correct).
* Class 2 is completely misclassified, with all 4 instances predicted as Class 0.
Overall, the model struggles to distinguish between classes, particularly Classes 1 and 2.
### 5) FIT RANDOM FOREST MODEL 
```{python}
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

IBD_model = RandomForestRegressor(random_state=1)
IBD_model.fit(train_X_std, train_y)
IBD_preds = IBD_model.predict(val_X_std)
print(mean_absolute_error(val_y, IBD_preds))
```
#### Visualisation MAE 
```{python}
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Calculate MAE
mae = mean_absolute_error(val_y, IBD_preds)

# Create a small table with MAE
data = {'Metric': ['Mean Absolute Error'], 'Value': [mae]}
mae_df = pd.DataFrame(data)

# Plot the table
fig, ax = plt.subplots(figsize=(4, 2))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
# Add the table
table = ax.table(cellText=mae_df.values, 
                 colLabels=mae_df.columns, 
                 loc='center', 
                 cellLoc='center', rowLoc='center')

# Adjust table styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title("Model 1: Mean Absolute Error", fontsize=10, weight='bold')
plt.tight_layout()
plt.show()
```
#### INTERPRETATION MAE (Mean Absolute Error) 
The Mean Absolute Error (MAE) = 0.667
* this indicates that, on average, the predicted class labels deviate by 0.67 units from the true class labels (where 0 = CD, 1 = UC, and 2 = nonIBD)
* This suggests that the model's predictions are not highly accurate, and there is room for improvement in classifying the conditions correctly
* Lower MAE values would indicate better model performance

## MACHINE LEARNING MODEL 2
Our first model did not perform well. For our next model, we will try to improve by using fewer features by excluding 'BMI_at_baseline', 'Height_at_baseline', and 'Weight_at_baseline'. This allows us to retain more data, as we no longer need to remove rows with missing values for these features.

### 1) Defining target value y and features X 
For our first machine learning model:
* our target value y = Study.Group_encoded
* our IBD_features (X) = interval_days, visit_num, Gender_encoded, Anitbiotics_encoded, Race_encoded
```{python}
#preparing the data for training a machine learning model by setting up the features (X) and the target variable (y)
y = metadata_adjusted['Study.Group_encoded']
IBD_features = ['interval_days',
       'visit_num', 'Gender_encoded',
       'Antibiotics_encoded', 'race_encoded]
X = metadata_adjusted [IBD_features]
```
```{python}
#get dimensions (number of rows, number of columns) of y 
y.shape
```
```{python}
#get dimensions (number of rows, number of columns) of X
X.shape
```
We see that we have 105 rows and 5 features that are represented as the columns 

### 2) Splitting data into train and test sets 
Then we split our dataset into training and test sets for building our machine learning model:
```{python}
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
```
### 3) PREPROCESSING DATA: Normalise feature values 
Next, we take a look at the distribution of our feature values, both for the train set and the test set
```{python}
import pandas as pd
import matplotlib.pyplot as plt

import warnings;
warnings.filterwarnings('ignore');
```
```{python}
#plotting the distrubution of the train set 
plt.figure(figsize=(18,6)) #set the size of the plot
train_X.sample(5, axis="columns").boxplot()
plt.xticks(rotation=90) #rotate the gene names on the x-axis
plt.show() #show the plot
```
```{python}
#plotting the distribution of the test set 
plt.figure(figsize=(18,6)) #set the size of the plot
val_X.sample(5, axis="columns").boxplot()
plt.xticks(rotation=90) #rotate the gene names on the x-axis
plt.show() #show the plot
```
The features clearly have very different scales. It is good practice to normalize the range of the features in the dataset.

We can do this with the Scikit-learn object StandardScaler:
```{python}
from sklearn.preprocessing import StandardScaler

#First we create a StandardScaler object called scaler_std: 
scaler_std = StandardScaler()
```
```{python}
#We first compute the mean and standard deviation from the train set:
scaler_std.fit(train_X)

print("Means:")
print(scaler_std.mean_)
print("Variances:")
print(scaler_std.var_)
```
```{python}
#We then compute the mean and standard deviation from the test set:
scaler_std.fit(val_X)

print("Means:")
print(scaler_std.mean_)
print("Variances:")
print(scaler_std.var_)
```
Next, we normalize the features in the train and test set.

The scaler object has a function transform() that uses the means and variances computed by the fit() function to normalize the features and we can make a Pandas DataFrame with the original column names as follows:
```{python}
#normalising the features if the train set 
train_X_std = scaler_std.transform(train_X)
train_X_std = pd.DataFrame(train_X_std, columns=train_X.columns)

train_X_std
```
```{python}
# normalising the features of the test set 
val_X_std = scaler_std.transform(val_X)
val_X_std = pd.DataFrame(val_X_std, columns=train_X.columns)

val_X_std
```
We can now plot the normalized features:
```{python}
#plotting distribution of the features of the train set 
plt.figure(figsize=(18,6))
train_X_std.sample(5, axis="columns").boxplot()
plt.xticks(rotation=90)
plt.show()
```
```{python}
#plotting distribution of the features of the train set
plt.figure(figsize=(18,6))
val_X_std.sample(5, axis="columns").boxplot()
plt.xticks(rotation=90)
plt.show()
```
As we can see, the feature value distributions are now normalized for both the train set and the test set

### 4) FIT LOGISTIC REGRESSION MODEL 
Now we can fit a logistic regression model.
In Scikit-learn we first initiate a LogisticRegression mode ! 
Train the model on the normalized train set:
```{python}
from sklearn.linear_model import LogisticRegression

cls_std = LogisticRegression()
```

```{python}
# Import the LogisticRegression model
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
cls_std = LogisticRegression()

# Fit the model on the normalized train set (features train_X_std and target train_y)
cls_std.fit(train_X_std, train_y)

# Optionally, print the model's parameters to check if it trained successfully
print("Model coefficients: ", cls_std.coef_)
print("Model intercept: ", cls_std.intercept_)
```

```{python}
#visualise model coefficients and model interecpt in plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Feature names and model parameters
feature_names = train_X_std.columns  # Assuming train_X_std is a DataFrame
coefficients = cls_std.coef_
intercepts = cls_std.intercept_

# Convert coefficients to a DataFrame for better visualization
coeff_df = pd.DataFrame(coefficients, columns=feature_names)
coeff_df['Class'] = [f"Class {i}" for i in range(coefficients.shape[0])]

# Melt the DataFrame for easier plotting
coeff_melted = coeff_df.melt(id_vars='Class', var_name='Feature', value_name='Coefficient')

# Plot coefficients
plt.figure(figsize=(12, 8))
sns.barplot(data=coeff_melted, x='Feature', y='Coefficient', hue='Class')
plt.title("Logistic Regression Coefficients per Class")
plt.xticks(rotation=45, ha="right")
plt.axhline(0, color='grey', linestyle='--')  # Line at 0 for reference
plt.tight_layout()
plt.show()

# Display intercepts separately
plt.figure(figsize=(6, 4))
sns.barplot(x=[f"Class {i}" for i in range(len(intercepts))], y=intercepts, palette="muted")
plt.title("Logistic Regression Intercepts per Class")
plt.ylabel("Intercept Value")
plt.axhline(0, color='grey', linestyle='--')
plt.tight_layout()
plt.show()
```
#### INTERPRETATION MODEL PARAMETERS
1.	Coefficients:
The model coefficients indicate the relationship between each feature and the probability of belonging to one of the three classes (0 = CD, 1 = UC, 2 = nonIBD).
Features with larger absolute coefficient values have a stronger impact on the prediction. For example:
•	Class 0 (CD) shows relatively moderate positive coefficients for certain features (e.g., feature 3 has 0.392), suggesting its importance for predicting this class.
•	Class 1 (UC) has several negative coefficients (e.g., feature 1 at -0.338), indicating these features reduce the likelihood of belonging to this class.
•	Class 2 (nonIBD) has a mix of positive and negative coefficients, with feature 1 (0.238) being notable.
2.	Intercepts: The intercept values control the baseline prediction for each class when all features are zero. For example:
•	Class 0 has a higher intercept (0.465), meaning the model is initially more biased towards predicting this class compared to the others.
To determine quality of the model, metrics such as accuracy, precision, recall, or F1-score are needed, as well as insights from a confusion matrix. However, the spread and variability in coefficients suggest the model is using the features to distinguish between classes, which is a positive sign. That said, further evaluation is required to confirm if these coefficients result in accurate predictions.

Computing the accuracy: 
```{python}
# Compute predictions on the validation set val_X_std
predictions_std = cls_std.predict(val_X_std)

# Optionally, print the predicted class labels for each row in val_X_std
print(predictions_std)

# Alternatively, to get the class probabilities instead of predicted classes:
predictions_std_proba = cls_std.predict_proba(val_X_std)

# Display the first 10 predicted class probabilities (for both class 0 and class 1)
print(predictions_std_proba[:10])

# Compute the accuracy of the model's predictions on the validation set
accuracy = cls_std.score(val_X_std, val_y)
print("Accuracy: {}".format(accuracy))
```
#### INTERPRETATION ACCURACY
* The accuracy of this machine learning model is 44%, meaning the model correctly classified about 44% of the data points
* this low accuracy suggests that the model performs only slightly better than random guessing for a multi-class classification problem
* this also suggests that the model is struggling to generalize well, particularly in distinguishing between classes.

We calculate some other metrics to evaluate the model training
```{python}
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

precision = precision_score(val_y, predictions_std, average='weighted')
recall = recall_score(val_y, predictions_std, average='weighted')
f1 = f1_score(val_y, predictions_std, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Confusion Matrix
cm = confusion_matrix(val_y, predictions_std)
print("Confusion Matrix:\n", cm)

```
#### Visualisation other metrics 
```{python}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Example predictions (replace these with your actual data)
predictions_std = cls_std.predict(val_X_std)
accuracy = cls_std.score(val_X_std, val_y)
precision = precision_score(val_y, predictions_std, average='weighted')
recall = recall_score(val_y, predictions_std, average='weighted')
f1 = f1_score(val_y, predictions_std, average='weighted')
cm = confusion_matrix(val_y, predictions_std)

# Create a dataframe for metrics
data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
}
metrics_df = pd.DataFrame(data)
# Plot the table and confusion matrix
fig, ax = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 2]})

# Add the main title
fig.suptitle('Model 2', fontsize=14, fontweight='bold')

# Table for metrics
ax[0].axis('tight')
ax[0].axis('off')
table = ax[0].table(cellText=metrics_df.values,
                    colLabels=metrics_df.columns,
                    loc='center',
                    cellLoc='center', rowLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
# Confusion matrix heatmap
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues', ax=ax[1], colorbar=False)
ax[1].set_title('Confusion Matrix', fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the suptitle
plt.show()
```
#### INTERPRETATION OTHER METRICS
The model's performance is quite limited. 
•	The precision = 29.0% => indicates that only 29% of the predicted positive cases are correct. 
•	The recall = 44.4% => shows that the model identifies 44.4% of the actual positive cases. 
•	The F1-score = 32.3% => reflects a poor balance between precision and recall. 
•	The confusion matrix => shows that most predictions fall into the first class, with minimal success in identifying the other classes. 
This suggests the model struggles to distinguish between the target classes effectively.
### 5) FIT RANDOM FOREST MODEL 
```{python}
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

IBD_model = RandomForestRegressor(random_state=1)
IBD_model.fit(train_X_std, train_y)
IBD_preds = IBD_model.predict(val_X_std)
print(mean_absolute_error(val_y, IBD_preds))
```
#### Visualisation MAE 
```{python}
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Calculate MAE
mae = mean_absolute_error(val_y, IBD_preds)

# Create a small table with MAE
data = {'Metric': ['Mean Absolute Error'], 'Value': [mae]}
mae_df = pd.DataFrame(data)

# Plot the table
fig, ax = plt.subplots(figsize=(4, 2))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
# Add the table
table = ax.table(cellText=mae_df.values, 
                 colLabels=mae_df.columns, 
                 loc='center', 
                 cellLoc='center', rowLoc='center')

# Adjust table styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title("Model 2: Mean Absolute Error", fontsize=10, weight='bold')
plt.tight_layout()
plt.show()
```
#### INTERPRETATION MAE (Mean Absolute Error) 
The Mean Absolute Error (MAE) = 0.755 => indicates that, on average, the predictions of the Random Forest model deviate by approximately 0.76 from the true class labels. Given that the target classes are discrete (0 = CD, 1 = UC, 2 = nonIBD), this suggests the model has difficulty accurately predicting the correct class, often predicting a class that is "close" but not exact. 
This highlights a need for further improvement of the model 



## MACHINE LEARNING MODEL 3
Our second machine learning model did not show significant improvement, so we will attempt another iteration by including the feature 'fecalcal'. Although this feature contains a substantial number of missing values, it could potentially enhance the model's predictive performance.
Due to some outliers in the nonIBD study group in the column 'fecalcal' we firstly clear the dataset for these.
```{python}
import pandas as pd
import numpy as np

# Copy the dataset to work safely
metadata_clean = metadata_adjusted.copy()

# Step 1: Filter the nonIBD group and remove NAs
nonIBD_data = metadata_clean[metadata_clean['Study.Group'] == 'nonIBD']
nonIBD_data = nonIBD_data.dropna(subset=['fecalcal'])  # Remove NAs from fecalcal

# Calculate IQR for the 'fecalcal' column within the nonIBD group
Q1 = nonIBD_data['fecalcal'].quantile(0.25)
Q3 = nonIBD_data['fecalcal'].quantile(0.75)
IQR = Q3 - Q1

# Determine the bounds for outliers
Lower_Bound = Q1 - 1.5 * IQR
Upper_Bound = Q3 + 1.5 * IQR

# Filter out outliers for nonIBD
nonIBD_clean = nonIBD_data[
    (nonIBD_data['fecalcal'] >= Lower_Bound) & 
    (nonIBD_data['fecalcal'] <= Upper_Bound)
]
# Step 2: Retain the rows for UC and CD without filtering
UC_CD_data = metadata_clean[metadata_clean['Study.Group'].isin(['UC', 'CD'])]

# Step 3: Combine the filtered nonIBD data with the original UC and CD data
clean_metadata = pd.concat([nonIBD_clean, UC_CD_data], axis=0)

# Reset the index for neatness
clean_metadata = clean_metadata.reset_index(drop=True)

# Result
print("Number of rows before removing outliers:", len(metadata_clean))
print("Number of rows after removing outliers:", len(clean_metadata))

# Display the first few rows of the combined clean dataset
print(clean_metadata.head())
```
We transform the non-numeric values of the clean_metadata to numeric values
```{python}
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Apply the encoding
clean_metadata['Study.Group_encoded'] = label_encoder.fit_transform(clean_metadata['Study.Group'])

# View the mapping
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# View the result
print(clean_metadata[['Study.Group', 'Study.Group_encoded']].head())
```
```{python}
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
label_encoder = LabelEncoder()
```

```{python}
# Variables to encode
categorical_columns = ['Gender', 'Antibiotics', 'race']
```
```{python}
# Loop through the columns and apply label encoding
for column in categorical_columns:
    clean_metadata[f'{column}_encoded'] = label_encoder.fit_transform(clean_metadata[column])
    print(f"Mapping for {column}:")
    print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    print()  # Empty line for readability
```
```{python}
# View the result (for example, the first few rows)
print(clean_metadata[[col for col in categorical_columns] + [f'{col}_encoded' for col in categorical_columns]].head())
```
```{python}
#there are still some missing values in 'fecalcal' so we first remove these: 
clean_metadata_withoutNA = clean_metadata.dropna(subset=['fecalcal'], axis=0)
```
### 1) Defining target value y and features X 
For our first machine learning model:
* our target value y = Study.Group_encoded
* our IBD_features (X) = interval_days, visit_num, Gender_encoded, Anitbiotics_encoded, Race_encoded, fecalcal
```{python}
#preparing the data for training a machine learning model by setting up the features (X) and the target variable (y)
y = clean_metadata_withoutNA['Study.Group_encoded']
IBD_features = ['interval_days',
       'visit_num', 'Gender_encoded',
       'Antibiotics_encoded', 'race_encoded','fecalcal']
X = clean_metadata_withoutNA[IBD_features]
```
```{python}
#get dimensions (number of rows, number of columns) of y 
y.shape
```
```{python}
#get dimensions (number of rows, number of columns) of X
X.shape
```
We see that we have 78 remaining rows after removing outliers and missing values in 'fecalcal' and 6 features that are represented as the columns

### 2) Splitting data into train and test sets 
Then we split our dataset into training and test sets for building our machine learning model:
```{python}
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
```
### 3) PREPROCESSING DATA: Normalise feature values 
Next, we take a look at the distribution of our feature values, both for the train set and the test set
```{python}
import pandas as pd
import matplotlib.pyplot as plt

import warnings;
warnings.filterwarnings('ignore');
```
```{python}
#plotting the distrubution of the train set 
plt.figure(figsize=(18,6)) #set the size of the plot
train_X.sample(6, axis="columns").boxplot()
plt.xticks(rotation=90) #rotate the gene names on the x-axis
plt.show() #show the plot
```
```{python}
#plotting the distribution of the test set 
plt.figure(figsize=(18,6)) #set the size of the plot
val_X.sample(6, axis="columns").boxplot()
plt.xticks(rotation=90) #rotate the gene names on the x-axis
plt.show() #show the plot
```
The features clearly have very different scales. It is good practice to normalize the range of the features in the dataset.

We can do this with the Scikit-learn object StandardScaler:
```{python}
from sklearn.preprocessing import StandardScaler

#First we create a StandardScaler object called scaler_std: 
scaler_std = StandardScaler()
```
```{python}
#We first compute the mean and standard deviation from the train set:
scaler_std.fit(train_X)

print("Means:")
print(scaler_std.mean_)
print("Variances:")
print(scaler_std.var_)
```
```{python}
#We then compute the mean and standard deviation from the test set:
scaler_std.fit(val_X)

print("Means:")
print(scaler_std.mean_)
print("Variances:")
print(scaler_std.var_)
```
Next, we normalize the features in the train and test set.

The scaler object has a function transform() that uses the means and variances computed by the fit() function to normalize the features and we can make a Pandas DataFrame with the original column names as follows:
```{python}
#normalising the features if the train set 
train_X_std = scaler_std.transform(train_X)
train_X_std = pd.DataFrame(train_X_std, columns=train_X.columns)

train_X_std
```
```{python}
# normalising the features of the test set 
val_X_std = scaler_std.transform(val_X)
val_X_std = pd.DataFrame(val_X_std, columns=train_X.columns)

val_X_std
```
We can now plot the normalized features:
```{python}
#plotting distribution of the features of the train set 
plt.figure(figsize=(18,6))
train_X_std.sample(5, axis="columns").boxplot()
plt.xticks(rotation=90)
plt.show()
```
```{python}
#plotting distribution of the features of the train set
plt.figure(figsize=(18,6))
val_X_std.sample(5, axis="columns").boxplot()
plt.xticks(rotation=90)
plt.show()
```
As we can see, the feature value distributions are now normalized for both the train set and the test set

### 4) FIT LOGISTIC REGRESSION MODEL 
Now we can fit a logistic regression model.
In Scikit-learn we first initiate a LogisticRegression mode ! 
Train the model on the normalized train set:
```{python}
from sklearn.linear_model import LogisticRegression

cls_std = LogisticRegression()
```

```{python}
# Import the LogisticRegression model
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
cls_std = LogisticRegression()

# Fit the model on the normalized train set (features train_X_std and target train_y)
cls_std.fit(train_X_std, train_y)

# Optionally, print the model's parameters to check if it trained successfully
print("Model coefficients: ", cls_std.coef_)
print("Model intercept: ", cls_std.intercept_)
```

```{python}
#visualise model coefficients and model interecpt in plots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Feature names and model parameters
feature_names = train_X_std.columns  # Assuming train_X_std is a DataFrame
coefficients = cls_std.coef_
intercepts = cls_std.intercept_

# Convert coefficients to a DataFrame for better visualization
coeff_df = pd.DataFrame(coefficients, columns=feature_names)
coeff_df['Class'] = [f"Class {i}" for i in range(coefficients.shape[0])]

# Melt the DataFrame for easier plotting
coeff_melted = coeff_df.melt(id_vars='Class', var_name='Feature', value_name='Coefficient')

# Plot coefficients
plt.figure(figsize=(12, 8))
sns.barplot(data=coeff_melted, x='Feature', y='Coefficient', hue='Class')
plt.title("Logistic Regression Coefficients per Class")
plt.xticks(rotation=45, ha="right")
plt.axhline(0, color='grey', linestyle='--')  # Line at 0 for reference
plt.tight_layout()
plt.show()

# Display intercepts separately
plt.figure(figsize=(6, 4))
sns.barplot(x=[f"Class {i}" for i in range(len(intercepts))], y=intercepts, palette="muted")
plt.title("Logistic Regression Intercepts per Class")
plt.ylabel("Intercept Value")
plt.axhline(0, color='grey', linestyle='--')
plt.tight_layout()
plt.show()
```
#### INTERPRETATION MODEL PARAMETERS
**Interpretation of Model Parameters**
1) Coefficients:
The model coefficients indicate the influence of each feature on the predicted probability of each class (UC, CD, and nonIBD)

UC (Ulcerative Colitis):
* Positive coefficients for several features (0.010275, 0.18597852, 0.28646124, 0.34362554, 0.60718502) suggest that as these feature values increase, the likelihood of the sample being classified as UC increases.
* The feature with the highest coefficient (0.60718502) suggests a particularly strong influence on the prediction of UC.

CD (Crohn's Disease):
* Negative coefficients (-0.08684992, -0.1651958, -0.22918266) indicate that higher values for these features tend to decrease the likelihood of being classified as CD.
* However, some positive coefficients (0.34670619, 0.60680137) show that other features can increase the likelihood of CD.
* Conversely, a few positive coefficients (0.07657491, 0.26170075) increase the likelihood of nonIBD.

nonIBD:
* Negative coefficients (-0.63316743, -1.21398639) show that certain features significantly reduce the likelihood of the sample being classified as nonIBD.

Intercepts:
* UC: The intercept (0.58910607) suggests that, when all feature values are zero, the model's baseline prediction is higher for UC.
* CD: The intercept (0.31247959) indicates a slightly lower baseline likelihood of being classified as CD compared to UC.
* nonIBD: The negative intercept (-0.90158565) indicates that the baseline prediction for nonIBD is lower compared to UC and CD.

The model seems to have captured some meaningful relationships between the features and the study groups (UC, CD, and nonIBD). Some features, especially those with higher coefficients (e.g., 0.60718502 for UC), suggest that the model has strong predictors for certain classes. The negative coefficients for some features in nonIBD (-1.21398639) indicate the model is able to distinguish between the groups.

However, the relative size of the coefficients and intercepts suggests that the model could benefit from additional tuning or feature engineering to improve predictions. The magnitude of some coefficients implies certain features strongly influence class predictions, but a balanced impact across features and regularization could enhance the model's quality.


Computing the accuracy: 
```{python}
# Compute predictions on the validation set val_X_std
predictions_std = cls_std.predict(val_X_std)

# Optionally, print the predicted class labels for each row in val_X_std
print(predictions_std)

# Alternatively, to get the class probabilities instead of predicted classes:
predictions_std_proba = cls_std.predict_proba(val_X_std)

# Display the first 10 predicted class probabilities (for both class 0 and class 1)
print(predictions_std_proba[:10])

# Compute the accuracy of the model's predictions on the validation set
accuracy = cls_std.score(val_X_std, val_y)
print("Accuracy: {}".format(accuracy))
```
#### INTERPRETATION ACCURACY
* Unfortunatly we see again a very low accuray of 45%, meaning that the model correctly classified about 44% of the data points
* this low accuracy suggests that the model performs only slightly better than random guessing for a multi-class classification problem
* this also suggests that the model is struggling to generalize well, particularly in distinguishing between classes.

We calculate some other metrics to evaluate the model training
```{python}
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

precision = precision_score(val_y, predictions_std, average='weighted')
recall = recall_score(val_y, predictions_std, average='weighted')
f1 = f1_score(val_y, predictions_std, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Confusion Matrix
cm = confusion_matrix(val_y, predictions_std)
print("Confusion Matrix:\n", cm)

```
#### Visualisation other metrics 
```{python}
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Example predictions (replace these with your actual data)
predictions_std = cls_std.predict(val_X_std)
accuracy = cls_std.score(val_X_std, val_y)
precision = precision_score(val_y, predictions_std, average='weighted')
recall = recall_score(val_y, predictions_std, average='weighted')
f1 = f1_score(val_y, predictions_std, average='weighted')
cm = confusion_matrix(val_y, predictions_std)

# Create a dataframe for metrics
data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
}
metrics_df = pd.DataFrame(data)
# Plot the table and confusion matrix
fig, ax = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [1, 2]})

# Add the main title
fig.suptitle('Model 3', fontsize=14, fontweight='bold')

# Table for metrics
ax[0].axis('tight')
ax[0].axis('off')
table = ax[0].table(cellText=metrics_df.values,
                    colLabels=metrics_df.columns,
                    loc='center',
                    cellLoc='center', rowLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
# Confusion matrix heatmap
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues', ax=ax[1], colorbar=False)
ax[1].set_title('Confusion Matrix', fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leaves space for the suptitle
plt.show()
```
#### INTERPRETATION OTHER METRICS
The model's performance metrics show moderate predictive ability
Precision = 0.48 =>  indicates that 48% of the instances predicted as a certain class (UC, CD, or nonIBD) were correct.
Recall = 0.45 =>  means that the model correctly identified 45% of the actual instances for the predicted classes.
F1-Score = 0.4575 =>  suggests a need for improvement in capturing all relevant instances while maintaining prediction accuracy.
The Confusion Matrix further shows that the model struggles with distinguishing between some classes, as indicated by the misclassified instances, particularly in the UC and nonIBD categories.

### 5) FIT RANDOM FOREST MODEL 
```{python}
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

IBD_model = RandomForestRegressor(random_state=1)
IBD_model.fit(train_X_std, train_y)
IBD_preds = IBD_model.predict(val_X_std)
print(mean_absolute_error(val_y, IBD_preds))
```
#### Visualisation MAE 
```{python}
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Calculate MAE
mae = mean_absolute_error(val_y, IBD_preds)

# Create a small table with MAE
data = {'Metric': ['Mean Absolute Error'], 'Value': [mae]}
mae_df = pd.DataFrame(data)

# Plot the table
fig, ax = plt.subplots(figsize=(4, 2))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
# Add the table
table = ax.table(cellText=mae_df.values, 
                 colLabels=mae_df.columns, 
                 loc='center', 
                 cellLoc='center', rowLoc='center')

# Adjust table styling
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title("Model 3: Mean Absolute Error", fontsize=10, weight='bold')
plt.tight_layout()
plt.show()
```
#### INTERPRETATION MAE (Mean Absolute Error) 
The Mean Absolute Error (MAE) = 0.688 -> indicates that the model's predictions deviate from the actual values by approximately 0.688 units. This value suggests moderate prediction accuracy, with room for improvement in reducing the error between predicted and actual outcomes.

