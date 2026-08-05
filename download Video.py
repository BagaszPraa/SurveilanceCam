import yt_dlp

# url = "https://www.youtube.com/watch?v=Amxg6m4UQuY"
url = "https://www.youtube.com/watch?v=ORQYtz-eEiQ"

ydl_opts = {
    "outtmpl": "video/video_test.%(ext)s",
    "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
    "merge_output_format": "mp4",
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])