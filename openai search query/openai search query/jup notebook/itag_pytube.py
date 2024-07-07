from pytube import YouTube  # !pip install pytube
from pytube.exceptions import RegexMatchError
from tqdm.auto import tqdm  # !pip install tqdm

# where to save
save_path = "./mp3"

# for i, row in tqdm(videos_meta):
    # url of video to be downloaded
url = f"https://youtu.be/{row['Video ID']}"

# try to create a YouTube vid object
try:
    yt = YouTube(url)
except RegexMatchError:
    print(f"RegexMatchError for '{url}'")
    #continue

itag = None
# we only want audio files
files = yt.streams.filter(only_audio=True)
for file in files:
    # from audio files we grab the first audio for mp4 (eg mp3)
    if file.mime_type == 'audio/mp4':
        itag = file.itag
        break
if itag is None:
    # just incase no MP3 audio is found (shouldn't happen)
    print("NO MP3 AUDIO FOUND")
    #continue

# get the correct mp3 'stream'
stream = yt.streams.get_by_itag(itag)
# downloading the audio
stream.download(
    output_path=save_path,
    filename=f"{row['Video ID']}.mp3"
)
