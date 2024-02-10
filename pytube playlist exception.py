from pytube import Playlist
import os
pl = list(Playlist('https://www.youtube.com/playlist?list=PL4bZBI_tvM9ARGwggLYtY4dqT2hR4s2Mn'))
for p in pl:
    command=f"pytube {p}"
    try:
        os.system(command)
    except:
        print(p)
    
    
    
