# openai-sdk-exploration
## setup
- install dependencies: `pipenv install -r requirements.txt`
- fill in env. variables in `.env`
- for **book a meeting example** add fixed_args to `calendar_fixed_args.json`

## customer service example
how to run: `python richtech_customer_service.py`

## shopify example
how to run: `python shopify.py`

## book a meeting example
how to run: `python book_meeting.py`

- example conversation:
```text
You: I want to book a demo with ADAM
getAPIBotResponse Time: 3.3625450134277344
Bot: Great! I can help you with that. I'll need a few details to schedule the demo:

1. Your email address.
2. Your preferred timezone (e.g., America/New_York, Europe/London).
3. Your preferred date and time for the demo.

Once I have that information, I can book it for you!
You:  mjg2290@columbia.edu, New York, 2pm 6/19, 2025
Event created: https://www.google.com/calendar/event?eid=bGttZWUxMGRhdjk0bHQyY2Znc2NycGQ0M2cgYm90QGFya2xleC5haQ
getAPIBotResponse Time: 3.5661678314208984
Bot: Your demo with ADAM has been successfully booked for June 19, 2025, at 2 PM New York time. 

If you have any other questions or need further assistance, feel free to ask!  

```
## Attempt at using `FileSearchTool` for RAG
The `FileSearchTool` allows retrieving information from your OpenAI Vector Stores. Attempted to use this for RAG but it did not work very well. All files need to be downloaded locally or already in desired format(e.g., txt, pdf, md, etc.) online. No built in webscrapping. 
This is how you run it: `cd openai_sdk_RAG_test`, `python openai-sdk-RAG-discovery.py` 

