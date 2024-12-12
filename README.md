# PG19-group-project-MICROBIOME

# Research question 1 (univariate analysis): Can calprotectin levels be used as a diagnostic biomarker to differentiate between non-IBD, ulcerative colitis (UC), and Crohn's disease (CD) in patients?

**Null hypothesis:** There is no significant difference in fecal calprotectin levels between patients with CD (Crohn's Disease), UC (Ulcerative Colitis), and non-IBD (individuals without inflammatory bowel diseases).

**Alternative hypothesis:** There is a significant difference in fecal calprotectin levels between patients with CD (Crohn's Disease), UC (Ulcerative Colitis), and non-IBD (individuals without inflammatory bowel diseases).

## Data preparation:

Read data:
```{r}
metadata <- read_tsv('metadata.tsv')
```

Do we have missing values, and if so, in which columns?
```{r}
sum(is.na(metadata))
colnames(metadata)[colSums(is.na(metadata))>0]
```
We have 749 missing values in the next columns: 'consent_age','Age at diagnosis', 'fecalcal', 'BMI_at_baseline', Height_at_baseline', 'Weight_at_baseline' and 'smoking status'.

Amount of missing values for each column:
```{r}
col_na_count <- colSums(is.na(metadata))
barplot(col_na_count, main = "Amount of missing values for each column", col = "lightblue", names.arg = colnames(metadata), las = 2, cex.names = 0.5)
```

_hoe hebben we deze NA's gezien en waarom enkel in fecalcal verwijderen?_
Fecal calprotectin contains NA values, we will remove them.
```{r}
clean_data1 <- metadata %>% filter(!is.na(fecalcal)) 
print(clean_data1) 
```
Checking if all the NA values are removed.
```{r}
sum(is.na(clean_data1$fecalcal)) 
```
It is important to examine how the NA values are distributed across the three study groups. This helps to understand whether the missing data is randomly distributed or if there are systematic patterns. For example, if the NA values are concentrated in specific study groups, it may indicate issues in data collection or bias, which could impact the validity of further analyses
```{r} 
na_distribution <- metadata %>% 
group_by(Study.Group) %>%  
summarise( 
  Total_NA = sum(is.na(fecalcal)), 
  Percentage_NA = (Total_NA / n()) * 100
) 
print(na_distribution) 
``` 
We have examined the distribution of missing values, and percentage-wise, there is no notable difference between the three groups.

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
There is no real difference between the groups. Next step is to conduct a statistical test to determine if there is indeed no significant difference in the NA distribution across the study groups. 

Since we are working with categorical variables and need to examine the association between them, we will use a chi-square test, which assumes that the distribution between the groups is random.

The goal is actually to prove the null hypothesis, not to reject it: p > 0.05.
```{r} 
chisq_test_data <- metadata %>% 
  mutate(fecalcal_missing = ifelse(is.na(fecalcal), "Missing", "Not Missing")) %>% 
  count(Study.Group, fecalcal_missing) %>% 
  pivot_wider(names_from = fecalcal_missing, values_from = n, values_fill = 0) 

chisq_result <- chisq.test(chisq_test_data[,-1])  #Disregard the Study.Group column
print(chisq_result) 
```

### Interpretation: 
Based on the p-value (0.8778), we cannot reject the null hypothesis (H₀). This means that there is no evidence of a significant difference in the number of missing values (NAs) between the different study groups (CD, UC, non-IBD).

The missing values appear to be randomly distributed across the study groups. Therefore, in this case, you can assume that the missing data is likely missing completely at random (MCAR), meaning that the distribution of NAs does not show any systematic pattern and does not introduce bias into your analysis.

Now that the NA distribution has been checked, it is also important to verify whether there are any duplicate subjects, as these could potentially influence the results.
```{r} 
duplicated_subjects <- clean_data[duplicated(clean_data$Subject), ]
```
_waarom staat er in de code hierboven een komma voor ]?_

Retrieving the duplicate subjects:
```{r} 
cat("Aantal dubbele subjects:", nrow(duplicated_subjects), "\n")
print(duplicated_subjects) 
```

Duplicate subjects are removed, and the mean of the numerical data for these subjects is calculated while ignoring the NA values.
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

Controlling if the duplicate subjects have been removed:
```{r}
duplicated_subjects1 <- metadata_clean[duplicated(metadata_clean$Subject), ] 
cat("dubble subjects:", nrow(duplicated_subjects1), "\n") 
print(duplicated_subjects1) 
```
_again, waarom staat er in de code hierboven een komma voor ]?_

## Descriptive statistics:
_is deze tussentitel een goede vertaling voor 'Beschrijvende statistieken?_
Calculates the descriptive statistics (mean, median, SD, min, max) of fecalcal_mean per Study.Group.
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

### Interpretation:
We observe that the difference in the average values between CD and UC is relatively small. However, there is a much larger difference in the averages between both CD and non-IBD, as well as UC and non-IBD. Additionally, the minimum values across the study groups are very similar. In contrast, the maximum values are more spread out, except for between UC and CD. Notably, the maximum value for non-IBD is close to the median values of both UC and CD

Visualisation of the constribution:
```{r} 
ggplot(metadata_clean, aes(x = Study.Group, y = fecalcal_mean, fill = Study.Group)) + 
  geom_boxplot() + 
  geom_jitter(width = 0.2, alpha = 0.5) + 
  labs(title = "Calprotectine per Study group", y = "Calprotectine (µg/g)", x =      "Group") + 
  theme_minimal() 
``` 

Histogram of fecal calprotectine per group WITH outliers:
```{r} 
ggplot(metadata_clean, aes(x = fecalcal_mean, fill = Study.Group)) + 
  geom_histogram(binwidth = 50, alpha = 0.6, position = "identity") + 
  facet_wrap(~Study.Group) +  # Maak aparte plots per groep 
  labs(title = "Histogram per Groep", x = "Calprotectine (µg/g)", y = "Frequentie") + 
  theme_minimal() 
``` 

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

_in de word werd nu nomaals de berekening van de beschrijvende statistieken gedaan (mean,median,SD,min,max) wat dan exact hetzelfde resultaat geeft als hierboven, dus dit heb ik weggelaten_

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
  labs(title = "Fecalcal per Group (no outliers)", y = "Fecalcal (µg/g)", x = "Group") + 
  theme_minimal()
``` 

Histogram of fecal calprotectine per group WITHOUT outliers:
```{r} 
ggplot(clean_data_unique, aes(x = fecalcal_mean, fill = Study.Group)) + 
  geom_histogram(binwidth = 50, alpha = 0.6, position = "identity") + 
  facet_wrap(~Study.Group) +  
  labs(title = "Histogram per Groep (zonder outliers)", x = "Calprotectine (µg/g)", y = "Frequentie") + 
  theme_minimal()
```

### Interpretation:
We have added 5 new variables to the clean_data_unique dataset:
- Q1: The first quartile (25th percentile).
- Q3: The third quartile (75th percentile).
- IQR: The difference between Q3 and Q1.
- Lower Bound = Q1 − 1.5 ⋅ IQR
- Upper Bound = Q3 + 1.5 ⋅ IQR

After removing outliers, we are left with 78 subjects. In the histogram per group (without outliers) of clean_data_unique, we notice that the data do not follow a normal distribution, as no bell-shaped curve is observed per study group. Therefore, we will proceed with a Kruskal-Wallis test to analyze the differences between the groups.

Kruskal-Wallis test on the clean_data_unique (without outliers and duplicate subjects).
```{r} 
install.packages("coin") 
library(coin) 
kruskal.test(fecalcal_mean ~ Study.Group, data = clean_data_unique) 
``` 

### Interpretation:
There is a significant difference in fecal calprotectine levels across the three different study groups, with a standard asymptotic p-value of 0.0001005. We will now first calculate the exact p-value and then examine which specific groups differ using the Wilcoxon test as a post hoc analysis.

While the standard asymptotic p-value may not be optimal with a small number of observations, we have 77 observations in this case, making the p-value reliable. Therefore, we will proceed with calculating the exact p-value.

### Conclusion:
We can conclude that there is an extremely significant difference (p = 0.0001005) in the distribution of calprotectin concentration due to the three study groups.

Post-hoc analysis will be conducted using the pairwise Wilcoxon test to examine the differences between the groups.
```{r} 
pairwise_wilcox <- pairwise.wilcox.test( 
  x = clean_data_unique$fecalcal_mean, 
  g = clean_data_unique$Study.Group, 
  p.adjust.method = "bonferroni"  # Correctie voor multiple testing 
) 
print(pairwise_wilcox) 
``` 
### Interpretation:
The output displays the p-values from pairwise comparisons in a matrix format, with a Bonferroni correction applied for multiple testing. The results indicate that both CD and UC show significant differences in fecal calprotectin levels compared to non-IBD (p = 0.00014 and p = 0.00027, respectively). However, there is no significant difference in fecal calprotectin levels between CD and UC (p = 1.0000), meaning that fecal calprotectin cannot be used to distinguish between these two groups.

### Conclusion:
Fecal calprotectin can effectively differentiate non-IBD from both UC and CD, but it is not a reliable marker to distinguish between CD and UC.

Creating a boxplot using ggboxplot, with Wilcoxon p-values added to show statistical significance.
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

### Interpretation: 
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
install.packages("ggpubr") 
library(ggpubr) 
```
```{r} 
kruskal_combined <- kruskal.test(fecalcal_mean ~ Group_Combined, data = clean_data_unique) 
print(kruskal_combined) 
```

### Interpretation:
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

### Interpretation:
The 95% confidence intervals (CIs) for the median fecal calprotectin levels in the IBD and non-IBD groups show distinct patterns. For the IBD group, the CI ranges from 41.48 µg/g to 147.69 µg/g, indicating that the median calprotectin level is significantly elevated, with some variability in the data. In contrast, the non-IBD group has a narrower CI, from 14.87 µg/g to 22.72 µg/g, suggesting a more consistent distribution and much lower median levels compared to the IBD group.

Crucially, the non-overlapping CIs between these groups indicate a statistically significant difference in fecal calprotectin levels. This finding strongly supports the use of fecal calprotectin as a reliable biomarker to distinguish between individuals with IBD and those without. The elevated levels in the IBD group reflect underlying inflammation, further underscoring its diagnostic relevance.

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

### Interpretation:
The output provides the effect size (η²) of the Kruskal-Wallis test. A value of 0.229 (rounded) indicates that approximately 22.9% of the variance in calprotectin levels can be explained by the differences between the groups (IBD vs non-IBD). This suggests that the group classification has a notable impact on the variation observed in calprotectin levels.

### Conclusion:
The effect size indicates that the group classification ("IBD" vs "non-IBD") significantly contributes to the variation in fecal calprotectin levels, further supporting the utility of fecal calprotectin as a biomarker for distinguishing between these two groups.

---------------------------------------------------------------------------------------

# Research question 2 (multivariate analysis): Are bacterial classes associated with Non-IBD, Ulcerative Colitis (UC), and Crohn's Disease (CD) in terms of microbiome composition?

**Null hypothesis:** There is no significant difference in bacterial composition between the three study groups (class CD, UC and non-IBD).

**Alternative hypothesis:** There is significant difference in bacterial composition between the three study groups (class CD, UC and non-IBD).

## Data preparation:

Read data
```{r}
genera_counts <- read_tsv('genera.counts.tsv')
```

Do we have missing values, and if so, in which columns?
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
  return(NA)  # Geen match gevonden
}
classes <- sapply(column_names, extract_class_from_column_name)
unique_classes <- unique(classes)
cat("Unieke classes in de dataset:\n")
print(unique_classes)
```
In this, we see that we have 275 unique classes (the first class of output is counted as an NA). 

Now we are going to replace all bacterial column names with the bacterial class name:
 
```
column_names <- colnames(genera_counts)

# Functie om de klasse te extraheren uit de kolomnaam
extract_class_from_column_name <- function(column_name) {
  match <- regmatches(column_name, regexec("c__([A-Za-z0-9_-]+)", column_name))
  
  if (length(match[[1]]) > 1) {
    return(match[[1]][2])  # Return the class part
  }
  return(NA)  # No match found
}

# Verkrijg een vector van klassen voor elke kolom
classes <- sapply(column_names, extract_class_from_column_name)

# Verwijder NA's uit de unieke klassen
unique_classes <- unique(classes[!is.na(classes)])

cat("Unieke classes in de dataset:\n")
print(unique_classes)

# Functie om de kolomnamen te hernoemen zonder indexen
rename_columns_no_index <- function(column_names, classes) {
  new_column_names <- character(length(column_names))
  
  # Loop over de kolomnamen
  for (i in seq_along(column_names)) {
    class_name <- classes[i]
    
    # Alleen doorgaan als de class_name geldig is (niet NA)
    if (!is.na(class_name)) {
      new_column_names[i] <- class_name  # No index added, just class name
    } else {
      new_column_names[i] <- "Unknown"  # Als geen klasse is gevonden, noem de kolom "Unknown"
    }
  }
  
  return(new_column_names)
}

# Pas de functie toe om kolomnamen zonder indexen te hernoemen
new_column_names <- rename_columns_no_index(column_names, classes)

# Hernoem de kolommen in de dataset
colnames(genera_counts) <- new_column_names

# Bekijk de vernieuwde kolomnamen
print(colnames(genera_counts))
head(genera_counts)
```	
Sample is currently labeled as 'unknown'. Check if any other column is labeled as 'unknown' as well.

_hier nog code plaken._

Now, merge the columns with the same name. Additionally, rename the 'unknown' column back to 'samples'.
```
# Stap 1: Houd "Unknown" ongemoeid en hernoem naar "Sample"
genera_counts$Sample <- genera_counts$Unknown
genera_counts <- genera_counts[, names(genera_counts) != "Unknown"]

# Stap 2: Controleer en converteer alleen kolommen die numeriek zijn
genera_counts_numeric <- genera_counts
genera_counts_numeric[] <- lapply(genera_counts_numeric, function(col) {
  if (is.numeric(col)) {
    col # Als de kolom al numeriek is, laat deze ongemoeid
  } else if (is.factor(col) || is.character(col)) {
    suppressWarnings(as.numeric(col)) # Probeer tekst/factor om te zetten naar numeriek
  } else {
    NULL # Kolommen die niet kunnen worden geconverteerd, uitsluiten
  }
})

# Stap 3: Combineer kolommen met dezelfde naam door ze samen te voegen
genera_counts_combined <- as.data.frame(sapply(unique(names(genera_counts_numeric)), function(col) {
  if (col %in% names(genera_counts_numeric)) {
    rowSums(genera_counts_numeric[names(genera_counts_numeric) == col], na.rm = TRUE)
  } else {
    NULL
  }
}))

# Stap 4: Voeg de "Sample"-kolom terug toe aan de samengevoegde dataset
genera_counts_combined$Sample <- genera_counts$Sample

# Stap 5: Verplaats de "Sample"-kolom naar de eerste positie (optioneel)
genera_counts_combined <- genera_counts_combined[, c("Sample", setdiff(names(genera_counts_combined), "Sample"))]

# Stap 6: Controleer het resultaat
print(genera_counts_combined)
```
Append the 'Study.Group' column to the 'genera_counts_combined' dataset.

```
genera_counts_combined <- merge(genera_counts_combined, metadata[, c("Sample", "Study.Group")], by = "Sample")
```
Append the 'Subject' column to the 'genera_counts_combined' dataset.

```
genera_counts_combined <- merge(genera_counts_combined, metadata[, c("Sample", "Subject")], by = "Sample")
```


# Machine learning:

