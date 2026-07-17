from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id):
    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)

    return " ".join(item.text for item in transcript)