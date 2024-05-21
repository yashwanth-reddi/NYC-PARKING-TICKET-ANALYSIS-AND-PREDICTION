# Yahswanth Reddy Gundepally - G01391064
# Pushyami Bhagavathula 	  - G01356145

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, count, concat, lit, row_number, date_format
from pyspark.sql.functions import to_date, month, year, from_unixtime, unix_timestamp, monotonically_increasing_id, udf, current_date
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes, OneVsRest, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType
from pyspark.ml.stat import Correlation

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# for visualization


spark = SparkSession.builder.appName("example").getOrCreate()

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

df2019_2023 = spark.read.csv("/content/drive/MyDrive/Parking_Violations_Issued_-_Fiscal_Year_2019_20231205.csv", header=True, inferSchema=True)

df2020_2023 = spark.read.csv("/content/drive/MyDrive/Parking_Violations_Issued_-_Fiscal_Year_2020_20231205.csv", header=True, inferSchema=True)

df2021_2023 = spark.read.csv("/content/drive/MyDrive/Parking_Violations_Issued_-_Fiscal_Year_2021_20231204.csv", header=True, inferSchema=True)

df2022_2023 = spark.read.csv("/content/drive/MyDrive/Parking_Violations_Issued_-_Fiscal_Year_2022_20231204.csv", header=True, inferSchema=True)

df2023_2023 = spark.read.csv("/content/drive/MyDrive/Parking_Violations_Issued_-_Fiscal_Year_2023_20231204.csv", header=True, inferSchema=True)

df = df2019_2023.union(df2020_2023).union(df2021_2023).union(df2022_2023).union(df2023_2023).cache()


# df = spark.read.csv("/content/drive/MyDrive/20000rows.csv", header=True, inferSchema=True)
# ## temp


indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in ["Law Section","Plate ID","Violation Time","Sub Division","Plate Type", "Registration State", "Vehicle Body Type", "Vehicle Make", "Issuing Agency", "Violation County", "Street Name"]]

indexed_df = df
for indexer in indexers:
    indexed_df = indexer.fit(indexed_df).transform(indexed_df)

feature_columns = ["Summons Number", "Plate ID_index", "Registration State_index", "Plate Type_index", "Violation Code", "Vehicle Body Type_index", "Vehicle Make_index", "Issuing Agency_index", "Street Code1", "Street Code2", "Street Code3", "Vehicle Expiration Date", "Violation Precinct", "Issuer Precinct", "Issuer Code", "Violation Time_index", "Violation County_index", "Street Name_index", "Law Section_index", "Sub Division_index", "Vehicle Year", "Feet From Curb", "month", "year"]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_df = assembler.transform(indexed_df)

correlation_matrix = Correlation.corr(assembled_df, "features").head()

corr_matrix = correlation_matrix[0].toArray()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, xticklabels= feature_columns, yticklabels= feature_columns ,annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap")
plt.show()



df.show()

df = df.withColumn("Issue Date", to_date(col("Issue Date"), "MM/dd/yyyy"))
df = df.withColumn("month", month("Issue Date"))
df = df.withColumn("year", year("Issue Date"))
df = df.filter(col("year").isin([2020, 2021, 2022]))
df = df.withColumn("Day_Of_Week", date_format(col("Issue Date"), "E")).cache()
df = df.withColumn("Vehicle Expiration Date", to_date(col("Vehicle Expiration Date"), "yyyymmdd")).cache()
df = df.withColumn("Violation Time", concat(df["Violation Time"], lit("M")))
df = df.withColumn("Violation Time", from_unixtime(unix_timestamp(df["Violation Time"], "hhmma")).cast("timestamp"))
df = df.withColumn("Violation Time", date_format(df["Violation Time"], "HH"))

df.show()

dfcopy = df


values_to_remove = ['STX', 'NULL', '108', 'P', 'A', 'F', 'RICHM', 'ABX']

df_cleaned = df.filter(~F.col('Violation County').isin(values_to_remove)).cache()

replacement_mappings = {
    'King': 'K',
    'kings':'K',
    'KINGS':'K',
    'K F':'K',
    'KING': 'K',
    'Ks':'K',

    'Qns': 'Q',
    'QN':'Q',
    'QNS':'Q',
    'Qunees': 'Q',

    'Bronx': 'B',
    'BX':'B',
    'Bron': 'B',
    'BK':'B',

    'Rich': 'R',
    'RICH':'R',

    'MN': 'NY',
}


for from_str, to_str in replacement_mappings.items():
    df_cleaned = df_cleaned.withColumn('Violation County', F.regexp_replace('Violation County', from_str, to_str))


df = df_cleaned.select("Day_Of_Week","Violation Time","Issuer Precinct","Issuing Agency","Violation Code","Violation County")

df = df.withColumn("Violation Time", df["Violation Time"].cast("int"))

df = df.na.drop().cache()

df = StringIndexer(inputCol='Violation County', outputCol='target').fit(df).transform(df)
df = df.withColumn('target', col('target').cast("int"))





categorical_cols = ['Day_Of_Week','Issuing Agency']

indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid='keep') for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded") for col in categorical_cols]

feature_columns = [f"{col}_encoded" for col in categorical_cols] + ['Violation Time', 'Issuer Precinct', 'Violation Code']

assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

gbt_classifier = GBTClassifier(labelCol='target', featuresCol='features', maxIter=10, maxDepth=5, seed=42)
ovr_classifier = OneVsRest(classifier=gbt_classifier, labelCol='target')


classifiers = [
    RandomForestClassifier(labelCol='target', featuresCol='features', numTrees=10),
    DecisionTreeClassifier(labelCol='target', featuresCol='features', maxDepth=5),
    NaiveBayes(labelCol='target', featuresCol='features', smoothing=1.0, modelType='multinomial'),
    ovr_classifier
]

(training_data, testing_data) = df.randomSplit([0.8, 0.2])

results = {}

combined_predictions = testing_data.select('target').withColumn("id", monotonically_increasing_id())

for classifier in classifiers:
    pipeline = Pipeline(stages=indexers + encoders + [assembler, classifier])
    model = pipeline.fit(training_data)

    current_predictions = model.transform(testing_data).withColumn("id", monotonically_increasing_id())

    evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')
    accuracy = evaluator.evaluate(current_predictions)

    model_name = classifier.__class__.__name__

    print(model_name,"classifier accuracy",accuracy)

    results[model_name] = accuracy

    current_predictions = current_predictions.withColumnRenamed("prediction", str(model_name))

    combined_predictions = combined_predictions.crossJoin(current_predictions.select(str(model_name)))

combined_predictions = combined_predictions.drop("id")

combined_predictions.show()

classifiers = list(results.keys())
accuracy_values = list(results.values())


plt.figure(figsize=(10, 6))
plt.plot(classifiers, accuracy_values, marker='o', linestyle='-')
plt.title('Classifier Accuracy Comparison')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.ylim(0, 1)


df = dfcopy


def bargraph(dataframe,variable,colour,year):
  variable_temp = dataframe.groupBy(variable).count().orderBy(col("count").desc()).limit(30)

  plt.figure(figsize=(20, 8))
  variable_temp.toPandas().plot(kind="bar", x=variable, y="count", color=colour)
  plt.xlabel(variable)
  plt.ylabel("Summons Count")
  plt.title("Summons Count by "+ variable +" in year :"+ year)
  plt.xticks(rotation=45, ha="right")
  plt.show()

def piechart(dataframe, variable, year, chart_size=6):
  variable_temp = dataframe.groupBy(variable).count().orderBy(col("count").desc()).limit(10)
  plt.figure(figsize=(chart_size, chart_size))
  plt.pie(variable_temp.toPandas()['count'], labels=variable_temp.toPandas()[variable], autopct='%1.1f%%', startangle=140, radius=0.8)
  plt.title("Distribution of Summons by " + variable + " in year: " + str(year))
  plt.show()

def expired_percent(dataframe,year):
  total_Vehicle = dataframe.count()
  expired_Vehicle = dataframe.filter(dataframe['Vehicle Expiration Date']<dataframe['Issue Date']).count()

  percentage_expired = (expired_Vehicle / total_Vehicle) * 100

  print("Percentage of vehicles past expiration date in year : ",year," is",percentage_expired)


def plate_type(dataframe,year):
  print("\n\nVechiles count by plate type in year :",year)
  dataframe.groupBy("Plate Type").count().orderBy(col("count").desc()).show(7)


def analyze_time_series(df, year):
  time_series_summons = df.groupBy("Violation Time").agg(count("Summons Number").alias("Summons Number"))
  time_series_summons = time_series_summons.withColumn("Year", lit(year))
  time_series_summons = time_series_summons.filter(col("Violation Time").isNotNull())
  time_series_summons = time_series_summons.sort("Violation Time")

  return time_series_summons

def lineplot(dataframe, variable, year):
  variable_temp = dataframe.groupBy(variable).count().orderBy(variable).limit(30)
  variable_temp = variable_temp.withColumnRenamed("count", "Summons Number")
  time_series_summons = variable_temp.toPandas()
  time_series_summons['Year'] = year
  logical_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  time_series_summons['Day_Of_Week'] = pd.Categorical(time_series_summons['Day_Of_Week'], categories=logical_order, ordered=True)
  time_series_summons = time_series_summons.sort_values('Day_Of_Week')
  time_series_summons = time_series_summons.reset_index(drop=True)
  return time_series_summons


def plot_monthly_summons_count(df, year, color):
    monthly_counts = df.groupBy("month").agg(F.count("Summons Number").alias("SummonsCount")).orderBy("month")

    plt.plot(
        monthly_counts.select("month").rdd.flatMap(lambda x: x).collect(),
        monthly_counts.select("SummonsCount").rdd.flatMap(lambda x: x).collect(),
        label=str(year),
        color=color
    )

df = dfcopy

df2020 = df.filter(col("year").isin(2020)).cache()
df2021 = df.filter(col("year").isin(2021)).cache()
df2022 = df.filter(col("year").isin(2022)).cache()


print("\n\n")
expired_percent(df2020,"2020")
expired_percent(df2021,"2021")
expired_percent(df2022,"2022")


plate_type(df2020,"2020")
plate_type(df2021,"2021")
plate_type(df2022,"2022")



df2020.groupBy("Vehicle Make").count().orderBy(col("count").desc()).show()
df2021.groupBy("Vehicle Make").count().orderBy(col("count").desc()).show()
df2022.groupBy("Vehicle Make").count().orderBy(col("count").desc()).show()



df2020.groupBy("Violation Code").count().orderBy(col("count").desc()).show()
df2021.groupBy("Violation Code").count().orderBy(col("count").desc()).show()
df2022.groupBy("Violation Code").count().orderBy(col("count").desc()).show()



print("\n\nOver three years intereting correlation\n\n")

correlation_df = df.groupBy("Violation Code", "Vehicle Body Type", "Registration State").count().orderBy(col("count").desc())
correlation_df.show()

correlation_df = df.groupBy("Vehicle Make", "Violation Code").count().orderBy(col("count").desc())
correlation_df.show()

agency_counts = df.groupBy("Issuing Agency").count().orderBy(col("count").desc())
agency_counts.show()

street_violations = df.groupBy("Street Name", "Violation Code").count()
street_violations.show()

top_streets = df.groupBy("Street Name").count().orderBy(col("count").desc())
top_streets.show()

df = df.withColumn("Vehicle Year", when(col("Vehicle Year") == 0, 2023).otherwise(col("Vehicle Year")))
df = df.withColumn("Vehicle Age", 2023 - col("Vehicle Year"))
df.groupBy("Vehicle Age").count().orderBy(col("count").desc()).show()


bargraph(df2020,"Vehicle Body Type","blue","2020")
bargraph(df2021,"Vehicle Body Type","red","2021")
bargraph(df2022,"Vehicle Body Type","green","2022")


piechart(df2020,"Registration State","2020")
piechart(df2021,"Registration State","2021")
piechart(df2022,"Registration State","2022")


datetime_series_2020 = lineplot(df2020, "Day_Of_Week", 2020)
datetime_series_2021 = lineplot(df2021, "Day_Of_Week", 2021)
datetime_series_2022 = lineplot(df2022, "Day_Of_Week", 2022)


combined_time_series = pd.concat([datetime_series_2020, datetime_series_2021, datetime_series_2022])


plt.figure(figsize=(10, 6))
sns.lineplot(x="Day_Of_Week", y="Summons Number", hue="Year", data=combined_time_series, palette=['red', 'blue', 'green'])
sns.scatterplot(x="Day_Of_Week", y="Summons Number", hue="Year", data=combined_time_series, palette=['red', 'blue', 'green'])
plt.xlabel("Day of the Week")
plt.ylabel("Summons Count")
plt.title("Time Series Analysis of Parking Summons")
plt.xticks(rotation=90)
plt.show()


plot_monthly_summons_count(df2020, 2020, 'green')
plot_monthly_summons_count(df2021, 2021, 'blue')
plot_monthly_summons_count(df2022, 2022, 'red')


plt.title('Monthly Summons Count Over Three Years')
plt.xlabel('Month')
plt.ylabel('Summons Count')
plt.legend()
plt.grid(True)
plt.show()



time_series_2020 = analyze_time_series(df2020, 2020)
time_series_2021 = analyze_time_series(df2021, 2021)
time_series_2022 = analyze_time_series(df2022, 2022)


combined_time_series = pd.concat([ time_series_2020.toPandas(), time_series_2021.toPandas(), time_series_2022.toPandas()])

plt.figure(figsize=(10, 6))
sns.lineplot(x="Violation Time", y="Summons Number", hue="Year", data=combined_time_series, palette=['red', 'blue', 'green'])
plt.xlabel("Hour of day (24 hours format)")
plt.xticks(rotation=90)
plt.ylabel("Summons Count")
plt.title("Time Series Analysis of Parking Summons")
plt.show()

df = df.withColumn('year', F.year('Issue Date'))
df = df.withColumn('month', F.month('Issue Date'))
df = df.withColumn('day', F.dayofmonth('Issue Date'))


public_holidays = ['2021-01-01', '2021-07-04', '2021-09-02', '2021-11-28', '2021-12-25',
                    '2022-01-01', '2022-07-04', '2022-09-05', '2022-11-24', '2022-12-25',
                    '2023-01-01', '2023-07-04', '2023-09-04', '2023-11-23', '2023-12-25']

public_holidays_df = spark.createDataFrame(pd.DataFrame({'public_holiday': public_holidays}))


df_with_holidays = df.join(public_holidays_df, (df.year == F.year('public_holiday')) & (df.month == F.month('public_holiday')) & (df.day == F.dayofmonth('public_holiday')), 'left_outer')


violations_per_holiday = df_with_holidays.groupBy('year', 'public_holiday').agg(F.count('Summons Number').alias('num_violations'))

violations_per_holiday = violations_per_holiday.na.drop().cache()

violations_per_holiday = violations_per_holiday.orderBy('public_holiday')


violations_per_holiday_pd = violations_per_holiday.toPandas()


fig, ax = plt.subplots(figsize=(10, 6))


year_colors = {'2021': 'red', '2022': 'blue', '2023': 'green'}


for index, row in violations_per_holiday_pd.iterrows():
    color = year_colors.get(str(row['year']), 'gray')
    ax.bar(row['public_holiday'], row['num_violations'], color=color)

ax.set_xlabel('Public Holiday')
ax.set_ylabel('Number of Violations')
ax.set_title('Number of Violations on Public Holidays (2021-2023)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()


nyc_parking_tickets = df2023_2023.limit(100000).toPandas()

top_10_violations = nyc_parking_tickets.loc[:,['Violation Description','Summons Number']].groupby(['Violation Description']).count()['Summons Number'].reset_index().sort_values('Summons Number',ascending = False).head(10)

sns.barplot(x = 'Summons Number', y = 'Violation Description', data = top_10_violations)







