import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager, rc
import shutil

# 숫자 레이블을 문자열로 변환
label_map = {
    0: "CP",
    1: "Gate review",
    2: "BOM",
    3: "PFMEA",
    4: "DFMEA",
    5: "제품사양서",
    6: "도면",
    "전체 문서": "전체 문서"
}

# 전체 문서에 대한 정보를 담은 데이터 로드
def load_dataframe(data):
    emb_df = pd.read_pickle(data)

    return emb_df

class DocList:
    def __init__(self, data):
        # 레이블 문자열 변환 정보
        self.label_map = label_map
        # 데이터 경로
        self.data = data
        
        # 한글 폰트 설정
        self.font_path = "/data/LG_ES_POC/002. PoC_개발/50. 중간보고 시연/00. data/malgun.ttf"
        self.font_prop = font_manager.FontProperties(fname = self.font_path).get_name()
        
        # 표 작성 함수 실행
        self.counted_df = self.make_table()
    
    # 표 작성    
    def make_table(self):
        # 전체 문서에 대한 정보를 담은 데이터 로드
        emb_df = load_dataframe(self.data)
        # 전체 문서에 대해 자동 분류된 결과값 count한 데이터프레임 선언
        count_doc = emb_df['pred'].value_counts()

        # 인덱스 번호 재정렬
        counted_df = count_doc.reset_index()
        # 컬럼 명 지정
        counted_df.columns = ['category', '문서 개수']
        # 'category' 컬럼을 기준으로 정렬
        counted_df = counted_df.sort_values(by='category').reset_index(drop=True)

        # 전체 문서 개수 세기
        total_count = emb_df['pred'].count()
        # 전체 문서 개수에 대한 정보를 맨 아래 행에 새로 삽입
        counted_df.loc[counted_df.shape[0]] = ['전체 문서', total_count]
        
        # 숫자값으로 label되어있는 category를 문자열로 전환
        counted_df['문서 종류'] = counted_df['category'].map(self.label_map)
        # 필요 없어진 컬럼 삭제
        counted_df = counted_df.drop(columns='category')

        # 데이터프레임 모양 정리
        cols = counted_df.columns.tolist()
        a = cols.index('문서 개수')
        b = cols.index('문서 종류')
        cols.insert(a, cols.pop(b))

        counted_df = counted_df[cols]
        
        # 숫자에 쉼표 추가
        counted_df['문서 개수'] = counted_df['문서 개수'].apply(lambda x: "{:,.0f}".format(x))
        
        return counted_df
    
    # 표 시각화
    def draw(self):
        tb1 = plt.table(cellText = self.counted_df.values, colLabels = self.counted_df.columns, loc='center', cellLoc='right')
        for key , cell  in tb1.get_celld().items():
            cell._text.set_fontproperties(self.font_prop)
            if key[0] == 0:
                cell.set_facecolor('lightblue')
            if key[0] == 8:
                cell.set_facecolor('lightgray')
        tb1.auto_set_font_size(False)
        tb1.set_fontsize(18)

        cellDict = tb1.get_celld()
        for i in range(0, len(self.counted_df.columns)):
            for j in range(0, len(self.counted_df.values)+1):
                cellDict[(j,i)].set_height(0.088)

        plt.axis('off')
        plt.tight_layout()
        plt.show()