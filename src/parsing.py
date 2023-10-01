import csv
import re


def parse_file(file):
    # Split the chat history into lines
    lines = file.split("\n")

    # Prepare the CSV file
    with open("parsed_chat.csv", "w", encoding="utf-8", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing the column names
        csvwriter.writerow(["date", "time", "user", "text"])

        for line in lines:
            # Using regular expressions to match the date, time, user and text
            match = re.search(r"(\d+\.\d+\.\d+), (\d+:\d+) - (.*?): (.*)", line)

            if match:
                date, time, user, text = match.groups()
                csvwriter.writerow([date, time, user, text])
