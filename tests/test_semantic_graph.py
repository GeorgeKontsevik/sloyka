#import pandas as pd

#from sloyka import Semgraph


#sm = Semgraph()
#test_df  = pd.read_feather("sloyka/sample_data/processed/df_strts.feather")[:20]
#text_column='Текст комментария'
#toponim_column='only_full_street_name'
#toponim_name_column='initial_street'
#toponim_type_column='Toponims'

#def test_extract_keywords():
    #result = sm.extract_keywords(test_df,
                                 #text_column,
                                 #toponim_column,
                                 #toponim_name_column,
                                 #toponim_type_column,
                                 #semantic_key_filter=0.6,
                                 #top_n=5)

    #assert len(result) == 6

# def test_get_semantic_closeness():
#     df = pd.DataFrame([['TOPONIM_1', 'роза'], ['TOPONIM_2', 'куст']], columns=['toponims', 'words'])
#     result = sm.get_semantic_closeness(df,
#                                        column='words',
#                                        similaryty_filter=0.5)

#     check = round(float(result['SIMILARITY_SCORE'].iloc[0]), 3)

#     assert check == round(0.655513, 3)

# def test_build_semantic_graph():
#     result = sm.build_semantic_graph(test_df,
#                                      text_column,
#                                      toponim_column,
#                                      toponim_name_column,
#                                      toponim_type_column,
#                                      key_score_filter=0.4,
#                                      semantic_score_filter=0.6,
#                                      top_n=5)
    
#     assert len(result.edges) == 216
