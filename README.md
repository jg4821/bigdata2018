# bigdata2018

This repo contains the project for the Big Data course at NYU. 

The count.py, clean_data.py, and LOF.py are written in python Spark. 

count.py counts the number of nulls and computes percentage of nulls for each column in a given dataset. The null values are user-specified. By calling -getmerge on Hadoop streaming, the function returns a single csv file listing each column name, followed by the number of nulls in that column, and percentage of nulls for that column. 

clean_data.py returns a cleaned version of given dataset that does not contain any null values in any column.

LOF.py computes the LOF score for each instance in a dataset. A value close to 1 or less than 1 indicate a normal instance. 
