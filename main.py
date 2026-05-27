from youtube_transcript_api import YouTubeTranscriptApi

#Fetching the transcript from the API
yt_api = YouTubeTranscriptApi()
fetched_transcript = yt_api.fetch("PqVbypvxDto")

#We convert the fetched transcript to plain text
plain_text = " ".join(chunk.text for chunk in fetched_transcript)
print(plain_text)