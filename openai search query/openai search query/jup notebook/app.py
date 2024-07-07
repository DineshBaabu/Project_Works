import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
import logging
import math

PINECONE_KEY = "d21ea354-bcbd-49ce-b9c2-e62edadbac44"   #st.secrets["PINECONE_KEY"]  # app.pinecone.io
INDEX_ID = 'youtube-search'

st.markdown("<link rel='stylesheet' type='text/css' href='https://huggingface.co/spaces/jamescalam/ask-youtube/raw/main/styles.css'>", unsafe_allow_html=True)

@st.cache_resource
def init_pinecone():
    pinecone.init(api_key=PINECONE_KEY, environment="us-east-1-aws")
    return pinecone.Index(INDEX_ID)
    
@st.cache_resource
def init_retriever():
    return SentenceTransformer("multi-qa-mpnet-base-dot-v1")

def make_query(query, retriever, top_k=10, include_values=False, include_metadata=True, filter=None):
    xq = retriever.encode([query]).tolist()
    logging.info(f"Query: {query}")
    attempt = 0
    while attempt < 3:
        try:
            xc = st.session_state.index.query(
                xq,
                top_k=top_k,
                include_values=include_values,
                include_metadata=include_metadata,
                filter=filter
            )
            matches = xc['matches']
            break
        except:
            # force reload
            pinecone.init(api_key=PINECONE_KEY, environment="us-east-1-aws")
            st.session_state.index = pinecone.Index(INDEX_ID)
            attempt += 1
            matches = []
    if len(matches) == 0:
        logging.error(f"Query failed")
    return matches

st.session_state.index = init_pinecone()
model_id = "multi-qa-mpnet-base-dot-v1"
model = SentenceTransformer(model_id)
retriever = model
# retriever = init_retriever()

# def card(thumbnail: str, title: str, urls: list, contexts: list, starts: list, ends: list):
#     meta = [(e, s, u, c) for e, s, u, c in zip(ends, starts, urls, contexts)]
#     meta.sort(reverse=False)
#     text_content = []
#     current_start = 0
#     current_end = 0
#     for end, start, url, context in meta:
#         # reformat seconds to timestamp
#         time = start / 60
#         mins = f"0{int(time)}"[-2:]
#         secs = f"0{int(round((time - int(mins))*60, 0))}"[-2:]
#         timestamp = f"{mins}:{secs}"
#         if start < current_end and start > current_start:
#             # this means it is a continuation of the previous sentence
#             text_content[-1][0] = text_content[-1][0].split(context[:10])[0]
#             text_content.append([f"[{timestamp}] {context.capitalize()}", url])
#         else:
#             text_content.append(["xxLINEBREAKxx", ""])
#             text_content.append([f"[{timestamp}] {context}", url])
#         current_start = start
#         current_end = end
#     html_text = ""
#     for text, url in text_content:
#         if text == "xxLINEBREAKxx":
#             html_text += "<br>"
#         else:
#             html_text += f"<small><a href={url}>{text.strip()}... </a></small>"
#             print(text)
#     html = f"""
#     <div class="container-fluid">
#         <div class="row align-items-start">
#             <div class="col-md-4 col-sm-4">
#                 <div class="position-relative">
#                     <a href={urls[0]}><img src={thumbnail} class="img-fluid" style="width: 192px; height: 106px"></a>
#                 </div>
#             </div>
#             <div  class="col-md-8 col-sm-8">
#                 <h2>{title}</h2>
#             </div>
#         <div>
#             {html_text}
#     <br><br>
#     """
#     return st.markdown(html, unsafe_allow_html=True)

# channel_map = {
#     'James Briggs': 'UCv83tO5cePwHMt1952IVVHw',
#     'Daniel Bourke': 'UCr8O8l5cCX85Oem1d18EezQ',
#     'Yannic Kilcher': 'UCZHmQk67mSJgfCCTn7xBfew',
#     'AI Coffee Break with Letitia': 'UCobqgqE4i5Kf7wrxRxhToQA',
#     'sentdex': 'UCfzlCWGWYyIQ0aLC5w48gBQ'
# }
    
st.write("""
# YouTube Q&A on Operating Systems
""")

# st.info("""
# YouTube search built as [explained here](https://pinecone.io/learn/openai-whisper)!
# *The current search scope is limited to a few videos talking about ML, NLP, and vector search*. Add requests for channels to include in the [*Community* tab](https://huggingface.co/spaces/jamescalam/ask-youtube/discussions).
# """)

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)

query = st.text_input("Search!", "")

# with st.expander("Advanced Options"):
#     channel_options = st.multiselect(
#         'Channels to Search',
#         ['James Briggs', 'Daniel Bourke', 'Yannic Kilcher', 'AI Coffee Break with Letitia', 'sentdex'],
#         ['James Briggs', 'Daniel Bourke', 'Yannic Kilcher', 'AI Coffee Break with Letitia', 'sentdex']
#     )

if query != "":
    # channels = [channel_map[name] for name in channel_options]
    print(f"query: {query}")
    matches = make_query(
        query, retriever, top_k=5 #,
        # filter={
        #     'channel_id': {'$in': channels}
        # }
    )
    # for context in matches:
    #     print(context['metadata'])
    # video_name = []
    # start = []
    # end = []
    # text = []
    # for context in matches:
    #     video_name.append(context['id'])
    #     start.append(context['metadata']['start'])
    #     end.append(context['metadata']['end'])
    #     text.append(context['metadata']['text'])
    
    for context in matches:
        if context['score'] > 18:
            start = int(context['metadata']['start'])
            # end = math.ceil(context['metadata']['end'])
            # url = f"https://youtube.com/embed/{context['id'][6:17]}?start={start}&end={end}"
            # min = int(context['metadata']['start']) // 60
            # sec = int(context['metadata']['start']) % 60
            url = f"https://youtube.com/embed/{context['id'][6:17]}?start={start}"
            # url = f"https://youtube.com/watch?v={context['id'][6:17]}#t={min}m{sec}s"
            thumbnail=f"https://img.youtube.com/vi/{context['id'][6:17]}/maxresdefault.jpg"
            html = f"""
            <div class="container-fluid">
                <div class="col-md-4 col-sm-4">
                    <div class="position-relative">
                        <a href={url}><img src={thumbnail} class="img-fluid" style="width: 192px; height: 106px"></a>
                    </div>
                </div>
                <div class="row align-items-start">
                    <div  class="col-md-8 col-sm-8">
                        <h2>{context['id'][18:]}</h2>
                    </div>
                <div>
                    {context['metadata']['start']}
                    {context['metadata']['end']}
                    {context['metadata']['text']}
            <br><br>
            """
            st.markdown(html, unsafe_allow_html=True)
        else:
            new_query = query.replace(" ","+")
            thumbnail = "https://img.youtube.com/vi/aaaaaaaaaaa/maxresdefault.jpg"
            url = f"https://www.youtube.com/results?search_query={new_query}" 
            html = f"""
            <div class="container-fluid">
                <div  class="col-md-8 col-sm-8">
                        <h2>No results available.</h2>
                        <h3>For further clarifications, Press ↓
                </div>
                <div class="col-md-4 col-sm-4">
                    <div class="position-relative">
                        <a href={url}><img src={thumbnail} class="img-fluid" style="width: 192px; height: 106px"></a>
                    </div>
                </div>
                    {context['score']}
            <br><br>
        #     """
            st.markdown(html, unsafe_allow_html=True)
            break

    
    # print(matches)

    
    # results = {}
    # order = []
    # for context in matches:
    #     video_id = context['metadata']['url'].split('/')[-1]
    #     if video_id not in results:
    #         results[video_id] = {
    #             'title': context['metadata']['title'],
    #             'urls': [f"{context['metadata']['url']}?t={int(context['metadata']['start'])}"],
    #             'contexts': [context['metadata']['text']],
    #             'starts': [int(context['metadata']['start'])],
    #             'ends': [int(context['metadata']['end'])]
    #         }
    #         order.append(video_id)
    #     else:
    #         results[video_id]['urls'].append(
    #             f"{context['metadata']['url']}?t={int(context['metadata']['start'])}"
    #         )
    #         results[video_id]['contexts'].append(
    #             context['metadata']['text']
    #         )
    #         results[video_id]['starts'].append(int(context['metadata']['start']))
    #         results[video_id]['ends'].append(int(context['metadata']['end']))
    # # now display cards
    # for video_id in order:
    #     card(
    #         thumbnail=f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
    #         title=results[video_id]['title'],
    #         urls=results[video_id]['urls'],
    #         contexts=results[video_id]['contexts'],
    #         starts=results[video_id]['starts'],
    #         ends=results[video_id]['ends']
    #     )