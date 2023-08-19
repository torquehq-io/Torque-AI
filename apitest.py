import csv
import requests

# Define the API endpoint
API_ENDPOINT = 'https://analytics.vmukti.com/api/analytics'
# https://analyticsapi.vmukti.com/api/Analytics
# Read data from the CSV file
with open('Users_slab/test/crowd_counting_history/people_count_history.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Populate the data dictionary with values from each row
        data = {
            "cameradid": row['cameradid'],
            "sendtime": row['sendtime'],
            "imgurl": row['imgurl'],
            "an_id": int(row['an_id']),  # Convert to int if necessary
            "ImgCount": int(row['ImgCount'])  # Convert to int if necessary
        }

        # Send post request and print the response
        r = requests.post(url=API_ENDPOINT, data=data)
        pastebin_url = r.text
        print("The pastebin URL is: %s" % pastebin_url)


        # roi_x = 323
        # roi_y = 208
        # roi_width = 192
        # roi_height = 182

        # roi_x = 238
        # roi_y = 148
        # roi_width = 235
        # roi_height = 159