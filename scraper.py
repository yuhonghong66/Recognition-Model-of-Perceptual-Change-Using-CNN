# -*- coding: utf-8 -*-
import urllib

def scrape_image(url, save_path):
    try:
        img = urllib.urlopen(url)
        # saveData というバイナリデータを作成
        saveData = open(save_path, 'wb')
        # saveDataに取得した画像を書き込み
        saveData.write(img.read())
        saveData.close()
    except:
        raise Exception("Failed to save image at " + url)

