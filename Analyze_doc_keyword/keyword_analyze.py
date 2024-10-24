import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PdfImage import display_pdf_page_by_filename
import es_opensearch as eos
from matplotlib.font_manager import FontProperties
from pprint import pprint
from IPython.display import display
import re
import os
import datetime

pdf_lake_dir = '/data/lake/ensol/processed/'
index_name = 'doc_lake_index'

# 문서의 작성일 및 문서 유형 정보 불러오기
def get_format_savedate(path):
    with open(path, 'r') as fp:
        meta = fp.read()
    fm = re.search(r'Format   : \\[(.+)\\]\\nFormat Code', meta, flags=re.DOTALL).groups()[0]
    date = re.search(r'SaveDTM  : \\[(\\d{4}[/-]\\d{2}[/-]\\d{2}).+\\]', meta, flags=re.DOTALL).groups()[0]
    return fm, date

# 문서 검색 및 검색한 문서의 정보 출력
class Search_Document:
    def __init__(self, data):
        self.df = self.load_dataframe(data)
        self.doc_name, self.fm, self.date, self.text, self.file_name, self.queried_index = self.query_index()
    
    # 데이터 프레임 로드
    def load_dataframe(self, data):
        tfidf_df = pd.read_pickle(data)

        return tfidf_df
    
    # 데이터 프레임에서 문서 검색 및 정보 가져오기
    def query_index(self):
    
        # 검색할 문서 번호 지정
        queried_index = int(input("[검색된 문서 id] "))
        
        # 검색된 문서
        queried_doc = eos.search_doc(index_name, queried_index)

        # 검색된 파일명
        file_name = queried_doc['_source']['lake']['name']

        # 검색된 문서명
        doc_name = queried_doc['_source']['origin']['name']

        # 검색된 문서의 텍스트
        text = queried_doc['_source']['full_text']

        # 검색된 문서의 유형, 작성날짜
        filtered_path = '/data/lake/ensol/filtered_3/'

        sec=queried_doc['_source']['lake']['id'][1:3]
        fir=queried_doc['_source']['lake']['id'][3:5]
        doc_type = queried_doc['_source']['origin']['type']
        fm, date = get_format_savedate(os.path.join(filtered_path,fir,sec, queried_doc['_source']['lake']['id']+doc_type+'-info.txt'))

        return doc_name, fm, date, text, file_name, queried_index

    # 가져온 문서 정보 출력하기    
    def print_doc_info(self):
        # 검색된 문서의 정보 출력
        print('')
        print('[문서명] ', self.doc_name)
        print(f'[유형] {self.fm}  [작성일] {self.date}')
        #print('[자동분류] ', file_pred_value)
        print('')
        if self.file_name.split('.')[1] == 'xls' or self.file_name.split('.')[1] == 'xlsx':
            pprint(re.sub(r'(\\n|~\\*~|~\\^~)', '  ', self.text[410:910]))
        else:
            filename = self.file_name.split('.')[0] + '.pdf'
            pdf_dir1 = self.file_name.split('.')[0][3:5]
            pdf_dir2 = self.file_name.split('.')[0][1:3]
            pdf_directory = os.path.join(pdf_lake_dir, pdf_dir1, pdf_dir2)
            pprint(re.sub(r'\\n', '', self.text[410:710]))
            print('')
            display_pdf_page_by_filename(pdf_directory, filename, 1)
        print('')


# 워드 클라우드 생성
class Word_Cloud:
    def __init__(self, tfidf_df, queried_index):
        self.df_subset = self.make_wordcloud_df(tfidf_df, queried_index)
    
    # 워드클라우드용 한 개 행의 데이터 프레임 생성
    def make_wordcloud_df(self, tfidf_df, queried_index):
        if tfidf_df[tfidf_df['file_name'].str.startswith(f'{queried_index}')]['file_name'].str.startswith(f'{queried_index}').tolist()[0] is True:

            subset_row = tfidf_df[tfidf_df['file_name'].str.startswith(f'{queried_index}')].index.tolist()[0]
            df_subset = tfidf_df.iloc[subset_row : subset_row+1]
    
        else:
            print(f"키워드 정보가 존재하지 않는 문서 : {queried_index}")
            
        return df_subset
    
    # 워드 클라우드 생성 및 출력
    def word_cloud(self):
        # 워드 클라우드 (Unigram)
        for idx, row in self.df_subset.iterrows():
            # 상위 키워드 불러오기
            top_n_keywords = row['Top_Keywords_unigram_str'].split()[:]
            top_n_weights = row['TFIDF_sorted_unigram'][:]

            # 가중치 값이 0 이상인 키워드만 가중치와 매핑 및 선언
            keywords = {word: weight for word, weight in zip(top_n_keywords, top_n_weights) if weight > 0}

            font_path = "/data/LG_ES_POC/002. PoC_개발/50. 중간보고 시연/00. data/malgun.ttf"
            font_prop = FontProperties(fname=font_path)

            # 원형 마스크 생성
            x, y = np.ogrid[:1000, :1000]
            mask = (x - 500) ** 2 + (y - 500) ** 2 > 500 ** 2
            mask = 255 * mask.astype(int)

            # 워드 클라우드 생성 (원형)
            wordcloud_circle_1 = WordCloud(
            font_path=font_path,
            background_color='white',
            #contour_width=3,
            #contour_color='white',
            colormap='Set2',
            mask=mask,
            width=300,
            height=300,
            scale=3.0,
            max_font_size=1000,
            prefer_horizontal=1.0,
            mode="RGBA",
            max_words=40
            ).generate_from_frequencies(keywords)

        #======================================================================

        # 워드 클라우드 (Bigram)
        for idx, row in self.df_subset.iterrows():
            # 상위 키워드 불러오기
            top_n_keywords = row['Top_Keywords_bigram']
            top_n_weights = row['TFIDF_sorted_bigram']

            # 가중치 값이 0 이상인 키워드만 가중치와 매핑 및 선언
            keywords = {k: v for k, v in zip(top_n_keywords, top_n_weights) if v > 0}

            font_path = "/data/LG_ES_POC/002. PoC_개발/50. 중간보고 시연/00. data/malgun.ttf"
            font_prop = FontProperties(fname=font_path)

            # 원형 마스크 생성
            x, y = np.ogrid[:1000, :1000]
            mask = (x - 500) ** 2 + (y - 500) ** 2 > 500 ** 2
            mask = 255 * mask.astype(int)

            # 워드 클라우드 생성 (원형)
            wordcloud_circle_2 = WordCloud(
            font_path=font_path,
            background_color='white',
            #contour_width=3,
            #contour_color='white',
            colormap='Set2',
            mask=mask,
            width=300,
            height=300,
            scale=3.0,
            max_font_size=1000,
            prefer_horizontal=1.0,
            mode="RGBA",
            max_words=40
            ).generate_from_frequencies(keywords)

            #========================================================

            # 두 개의 워드클라우드 시각화

            fig, axes = plt.subplots(1, 2, figsize=(15, 7))

            axes[0].imshow(wordcloud_circle_1, interpolation='bilinear')
            axes[0].axis('off')
            axes[0].set_title('Unigram')

            axes[1].imshow(wordcloud_circle_2, interpolation='bilinear')
            axes[1].axis('off')
            axes[1].set_title('Bigram')

            plt.tight_layout()
            plt.show()

# 핵심 키워드 리스트 표 출력
class Keyword_List: 
    def __init__(self, tfidf_df, queried_index):
        self.df_subset = self.make_keyword_list_df(tfidf_df, queried_index)
        self.queried_index = queried_index

    # 키워드 리스트용 데이터 프레임 생성
    def make_keyword_list_df(self, tfidf_df, queried_index):
        if tfidf_df[tfidf_df['file_name'].str.startswith(f'{queried_index}')]['file_name'].str.startswith(f'{queried_index}').tolist()[0] is True:

            subset_row = tfidf_df[tfidf_df['file_name'].str.startswith(f'{queried_index}')].index.tolist()[0]
            df_subset = tfidf_df.iloc[subset_row : subset_row+1]

        else:
            print(f"키워드 정보가 존재하지 않는 문서 : {queried_index}")

        return df_subset
    
    # 키워드 리스트 표 생성 및 출력
    def keyword_list(self):
        # Unigram 표 생성
        df_unigram_list = self.df_subset[['TFIDF_sorted_unigram','Top_Keywords_unigram_str']].copy()

        df_unigram_list['Top_Keywords_unigram_str'] = df_unigram_list['Top_Keywords_unigram_str'].apply(lambda x: x.split())

        df_unigram_list['TFIDF_sorted_unigram'] =  df_unigram_list['TFIDF_sorted_unigram'].apply(lambda lst: [round(item, 2) for item in lst])

        # 키워드와 tfidf 값의 쌍을 만들어 새로운 데이터프레임에 저장
        rows = []
        for _, row in df_unigram_list.iterrows():
            keywords_unigram = row['Top_Keywords_unigram_str']
            tfidf_values_unigram = row['TFIDF_sorted_unigram']
            for k, v in zip(keywords_unigram, tfidf_values_unigram):
                rows.append({'주요 키워드': k, '가중치값': v})

        # 새로운 데이터프레임 생성
        df_mapped_unigram = pd.DataFrame(rows)

        today = datetime.date.today().strftime('Y%-m%-d%')
        
        # 해당 문서의 키워드 파일 저장
        dir_path = f'./00. data/keyword_csv/{today}/{self.queried_index}'
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, f'{self.queried_index}_unigram.csv')
        df_mapped_unigram.to_csv(file_path, encoding='utf-8', index=False)
        
        # 10개까지 리스트로 시각화
        view_df_unigram = df_mapped_unigram[0:10]
        
        #======================================================================================
        
        # Bigram 표 생성
        df_bigram_list = self.df_subset[['TFIDF_sorted_bigram','Top_Keywords_bigram']].copy()

        df_bigram_list['Top_Keywords_bigram'] = df_bigram_list['Top_Keywords_bigram']

        df_bigram_list['TFIDF_sorted_bigram'] =  df_bigram_list['TFIDF_sorted_bigram'].apply(lambda lst: [round(item, 2) for item in lst])

        # 키워드와 tfidf 값의 쌍을 만들어 새로운 데이터프레임에 저장
        rows = []
        for _, row in df_bigram_list.iterrows():
            keywords_bigram = row['Top_Keywords_bigram']
            tfidf_values_bigram = row['TFIDF_sorted_bigram']
            for k, v in zip(keywords_bigram, tfidf_values_bigram):
                rows.append({'주요 키워드': k, '가중치값': v})

        # 새로운 데이터프레임 생성
        df_mapped_bigram = pd.DataFrame(rows)

        file_path = os.path.join(dir_path, f'{self.queried_index}_bigram.csv')
        df_mapped_bigram.to_csv(file_path, encoding='utf-8', index=False)

        # 10개까지 리스트로 시각화
        view_df_bigram = df_mapped_bigram[0:10]
        
        #======================================================================================
        font_path = "/data/LG_ES_POC/002. PoC_개발/50. 중간보고 시연/00. data/malgun.ttf"
        font_prop = FontProperties(fname=font_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # tabel 1 생성
        tb1 = axes[0].table(cellText = view_df_unigram.values, colLabels = view_df_unigram.columns, loc='center', cellLoc='center')
        for key , cell  in tb1.get_celld().items():
                cell._text.set_fontproperties(font_prop)
                if key[0] == 0:
                    cell.set_facecolor('lightblue')
        tb1.auto_set_font_size(False)
        tb1.set_fontsize(18)

        cellDict = tb1.get_celld()
        for i in range(0, len(view_df_unigram.columns)):
            for j in range(0, len(view_df_unigram.values)+1):
                cellDict[(j,i)].set_height(0.088)

        axes[0].axis('off')
        axes[0].set_title('Unigram')
        #========================================================
        # tabel 2 생성
        tb2 = axes[1].table(cellText = view_df_bigram.values, colLabels = view_df_bigram.columns, loc='center', cellLoc='center')
        for key , cell  in tb2.get_celld().items():
                cell._text.set_fontproperties(font_prop)
                if key[0] == 0:
                    cell.set_facecolor('lightblue')
        tb2.auto_set_font_size(False)
        tb2.set_fontsize(18)

        cellDict = tb2.get_celld()
        for i in range(0, len(view_df_bigram.columns)):
            for j in range(0, len(view_df_bigram.values)+1):
                cellDict[(j,i)].set_height(0.088)

        axes[1].axis('off')
        axes[1].set_title('Bigram')

        #plt.axis('off')
        plt.tight_layout()
        plt.show()
