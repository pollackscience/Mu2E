#! /usr/bin/env python

import csv
def make_csv(mydict):
    with open('Mau10_bad_r_test_req.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key, value])
