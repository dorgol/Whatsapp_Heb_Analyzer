import csv
import re

def parse_file(file):
    # Split the chat history into lines
    lines = file.split("\n")

    # Determine the date format by checking the first non-empty line
    date_format = None
    for line in lines:
        if line:
            if re.match(r"\d+\.\d+\.\d+", line):
                date_format = "d.m.y"
            elif re.match(r"\d+/\d+/\d+", line):
                date_format = "m/d/y"
            break

    if not date_format:
        raise ValueError("Unable to determine date format")

    # Prepare the CSV file
    with open("parsed_chat.csv", "w", encoding="utf-8", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing the column names
        csvwriter.writerow(["date", "time", "user", "text"])

        for line in lines:
            if date_format == "d.m.y":
                pattern = re.compile(r"(\d+\.\d+\.\d+), (\d+:\d+) - (.*?): (.*)")
            else:
                pattern = re.compile(r"(\d+/\d+/\d+), (\d+:\d+) - (.*?): (.*)")

            match = pattern.search(line)

            if match:
                date, time, user, text = match.groups()
                csvwriter.writerow([date, time, user, text])
