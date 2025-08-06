from .pose_bone_frame_maker import IPoseFrameMaker
from ..processed_data import ProcessedData

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

font_path = ".\_internal\Font\KBO.ttf"

class PoseOverlayFrameMaker(IPoseFrameMaker):
    def __init__(self, frame_maker:IPoseFrameMaker):
        self.frame_maker = frame_maker

    def set_data(self, data:ProcessedData, df:pd.DataFrame):
        if data is None: return

        self.frame_maker.set_data(data, df)
        self.df = df
        
    def get_img_at(self, idx:int):
        
        img = self.frame_maker.get_img_at(idx)

        if img is None:
            return None
        
        ret_img = self.overlay_df_data(
            img, self.df.iloc[idx].to_dict(), idx)

        return ret_img

    def overlay_df_data(self, img, data:dict, frame_cnt:int):

        ret_img = img.copy()

        ret_img, t = self.put_text(ret_img, str(frame_cnt),
                                (10, 10), 24,
                                (255, 255, 255),
                                (32, 32, 32, 0),
                                font_path)

        print_data = {}

        for name, value in data.items():
            if name in self.labels:
                print_data[name] = value

        y_offset = ret_img.shape[0] - (len(print_data) * 30)
        
        for i, (name, value) in enumerate(print_data.items()):

            pos = (10, y_offset + i * 30)

            ret_img, bg_box = self.put_text(ret_img, name,
                                 pos, 12,
                                 (255, 255, 255),
                                 (103, 153, 250, 255),
                                 font_path)
                                 
            ret_img, t = self.put_text(ret_img, str(value),
                                 (bg_box[2] + 5, pos[1]), 12,
                                 (255, 255, 255),
                                 (0, 0, 0, 128),
                                 font_path)
        
        return ret_img

    
    def put_text(self, image, text, pos, fontsize=24,
                 font_color=(0, 0, 0),
                 background_color=(255, 255, 255, 128),
                 font_path='font.ttf'):
        
        # numpy 배열을 PIL 이미지로 변환
        img_pil = Image.fromarray(image)
        
        # 투명도를 적용하기 위해 RGBA 모드로 변환합니다.
        # 원본 이미지의 크기를 유지하며 알파 채널을 추가합니다.
        img_rgba = img_pil.convert('RGBA')
        draw = ImageDraw.Draw(img_rgba)

        # 폰트 객체 생성
        try:
            font = ImageFont.truetype(font_path, fontsize)
        except IOError:
            print(f"Error: The font file '{font_path}' could not be loaded.")
            font = ImageFont.load_default()

        # 텍스트의 바운딩 박스(좌표) 계산
        bbox = draw.textbbox(pos, text, font=font)
        bg_box = (bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5)

        # 투명한 배경을 그리기 위한 새 레이어(이미지)를 생성합니다.
        # 이 레이어는 배경색을 채우고 투명도를 조절할 수 있게 해줍니다.
        # 기존 이미지와 동일한 크기와 RGBA 모드로 만듭니다.
        bg_layer = Image.new('RGBA', img_rgba.size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg_layer)

        # 투명한 배경을 새 레이어에 그립니다.
        bg_draw.rectangle(bg_box, fill=background_color)
        
        # 배경 레이어를 원본 이미지에 합성합니다.
        # Image.alpha_composite를 사용하면 투명도가 제대로 적용됩니다.
        final_rgba = Image.alpha_composite(img_rgba, bg_layer)
        
        # 텍스트를 최종 RGBA 이미지에 그립니다.
        # draw 객체를 final_rgba 이미지에 연결
        text_draw = ImageDraw.Draw(final_rgba)
        text_draw.text(pos, text, font=font, fill=font_color)

        # 최종 이미지를 RGB 모드로 변환하고 numpy 배열로 반환합니다.
        result_image = np.array(final_rgba.convert('RGB'))

        return result_image, bg_box