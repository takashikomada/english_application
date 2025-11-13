"""
===============================================================================
【更新履歴：回答精度向上アップデート（2025/11）】

本プロジェクトでは、シャドーイング／ディクテーションにおける
「評価の精度・一貫性向上」を目的に、以下の修正を実施しました。

■ 1. 評価専用 LLM（eval_llm）を追加
   - 採点処理専用に ChatOpenAI(model="gpt-4o-mini", temperature=0.0) を導入。
   - 会話用 LLM（temperature=0.5）と分離することで、
     採点時のブレを減らし、一貫した評価が可能に。

■ 2. 採点ロジックを全面的に刷新
   - これまでの「ConversationChain + memory」を利用した評価方式を廃止。
   - 各問題ごとに SYSTEM_TEMPLATE_EVALUATION を format した内容を、
     毎回 eval_llm に直接投げる方式へ変更（プロンプト直呼び）。
   - 過去会話や他モードの履歴に影響されない「純粋な 1 問ごとの採点」を実現。

■ 3. シャドーイング／ディクテーションの評価処理を統一化
   - どちらのモードでも評価処理は create_evaluation(system_template) を使用。
   - system_template には「問題文」と「ユーザー回答」を明示的に埋め込み、
     毎回独立した採点プロンプトとして処理。

■ 4. 不要フラグ（shadowing_evaluation_first_flg）を削除
   - 評価用Chainを使い回す仕様は撤廃したため、初回フラグを削除。

■ 5. 結果として向上した点
   - 採点が過去会話・前問題の影響を受けなくなり、精度が安定。
   - 同じ回答には同じ傾向の評価が返るため、学習体験の信頼性が向上。
   - コードの可読性と保守性も改善。

===============================================================================
"""

import streamlit as st
import os
import time
from time import sleep
from pathlib import Path
from streamlit.components.v1 import html
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import functions as ft
import constants as ct


# 各種設定
load_dotenv()
st.set_page_config(
    page_title=ct.APP_NAME
)

# タイトル表示
st.markdown(f"## {ct.APP_NAME}")

# ==== LLMまわりのセッション初期化（Cloud対策） ====
if "openai_obj" not in st.session_state:
    st.session_state.openai_obj = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=st.session_state.llm,
        max_token_limit=1000,
        return_messages=True
    )

if "chain_basic_conversation" not in st.session_state:
    # モード「日常英会話」用のChain作成
    st.session_state.chain_basic_conversation = ft.create_chain(
        ct.SYSTEM_TEMPLATE_BASIC_CONVERSATION
    )
# =============================================

# 初期処理
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.start_flg = False
    st.session_state.pre_mode = ""
    st.session_state.shadowing_flg = False
    st.session_state.shadowing_button_flg = False
    st.session_state.shadowing_count = 0
    st.session_state.shadowing_first_flg = True
    st.session_state.shadowing_audio_input_flg = False
    st.session_state.dictation_flg = False
    st.session_state.dictation_button_flg = False
    st.session_state.dictation_count = 0
    st.session_state.dictation_first_flg = True
    st.session_state.dictation_chat_message = ""
    st.session_state.chat_open_flg = False
    st.session_state.problem = ""

    # OpenAI / LLM の初期化
    st.session_state.openai_obj = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # 会話用（少しゆらぎあり）
    st.session_state.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    # 評価用（できるだけブレを抑える）
    st.session_state.eval_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=st.session_state.llm,
        max_token_limit=1000,
        return_messages=True,
    )

    # モード「日常英会話」用のChain作成
    st.session_state.chain_basic_conversation = ft.create_chain(
        ct.SYSTEM_TEMPLATE_BASIC_CONVERSATION
    )

# 初期表示
# col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
# 提出課題用
col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
with col1:
    if st.session_state.start_flg:
        st.button("開始", use_container_width=True, type="primary")
    else:
        st.session_state.start_flg = st.button(
            "開始", use_container_width=True, type="primary"
        )

# ★ ここから追加（開始の右側に「一旦中断」ボタン）
with col1:
    if st.button("一旦中断", use_container_width=True):
        st.session_state.start_flg = False
        st.session_state.shadowing_flg = False
        st.session_state.dictation_flg = False
        st.session_state.shadowing_audio_input_flg = False
        st.session_state.chat_open_flg = False
        st.stop()
# ★ 追加ここまで


with col2:
    st.session_state.speed = st.selectbox(
        label="再生速度",
        options=ct.PLAY_SPEED_OPTION,
        index=3,
        label_visibility="collapsed",
    )
with col3:
    st.session_state.mode = st.selectbox(
        label="モード",
        options=[ct.MODE_1, ct.MODE_2, ct.MODE_3],
        label_visibility="collapsed",
    )
    # モードを変更した際の処理
    if st.session_state.mode != st.session_state.pre_mode:
        # 自動でそのモードの処理が実行されないようにする
        st.session_state.start_flg = False
        # 「日常英会話」選択時の初期化処理
        if st.session_state.mode == ct.MODE_1:
            st.session_state.dictation_flg = False
            st.session_state.shadowing_flg = False
        # 「シャドーイング」選択時の初期化処理
        st.session_state.shadowing_count = 0
        if st.session_state.mode == ct.MODE_2:
            st.session_state.dictation_flg = False
        # 「ディクテーション」選択時の初期化処理
        st.session_state.dictation_count = 0
        if st.session_state.mode == ct.MODE_3:
            st.session_state.shadowing_flg = False
        # チャット入力欄を非表示にする
        st.session_state.chat_open_flg = False
    st.session_state.pre_mode = st.session_state.mode
with col4:
    st.session_state.englv = st.selectbox(
        label="英語レベル",
        options=ct.ENGLISH_LEVEL_OPTION,
        label_visibility="collapsed",
    )

with st.chat_message("assistant", avatar="images/ai_icon.jpg"):
    st.markdown(
        "こちらは生成AIによる音声英会話の練習アプリです。何度も繰り返し練習し、英語力をアップさせましょう。"
    )
    st.markdown("**【操作説明】**")
    st.success(
        """
    - モードと再生速度を選択し、「英会話開始」ボタンを押して英会話を始めましょう。
    - モードは「日常英会話」「シャドーイング」「ディクテーション」から選べます。
    - 発話後、5秒間沈黙することで音声入力が完了します。
    - 「一時中断」ボタンを押すことで、英会話を一時中断できます。
    """
    )
st.divider()

# メッセージリストの一覧表示
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="images/ai_icon.jpg"):
            st.markdown(message["content"])
    elif message["role"] == "user":
        with st.chat_message(message["role"], avatar="images/user_icon.jpg"):
            st.markdown(message["content"])
    else:
        st.divider()

# LLMレスポンスの下部にモード実行のボタン表示
if st.session_state.shadowing_flg:
    st.session_state.shadowing_button_flg = st.button("シャドーイング開始")
if st.session_state.dictation_flg:
    st.session_state.dictation_button_flg = st.button("ディクテーション開始")

# 🔊 ディクテーション用の音声プレイヤー（rerun 対策）
if (
    "dictation_audio_bytes" in st.session_state
    and st.session_state.mode == ct.MODE_3  # ディクテーションのときだけ表示
):
    st.audio(st.session_state.dictation_audio_bytes, format="audio/wav")

# 「ディクテーション」モードのチャット入力受付時に実行
if st.session_state.chat_open_flg:
    st.info(
        "AIが読み上げた音声を、画面下部のチャット欄からそのまま入力・送信してください。"
    )

st.session_state.dictation_chat_message = st.chat_input(
    "※「ディクテーション」選択時以外は送信不可"
)

if st.session_state.dictation_chat_message and not st.session_state.chat_open_flg:
    st.stop()

# 「英会話開始」ボタンが押された場合の処理
if st.session_state.start_flg:

    # モード：「ディクテーション」
    # 「ディクテーション」ボタン押下時か、「英会話開始」ボタン押下時か、チャット送信時
    if (
        st.session_state.mode == ct.MODE_3
        and (
            st.session_state.dictation_button_flg
            or st.session_state.dictation_count == 0
            or st.session_state.dictation_chat_message
        )
    ):
        if st.session_state.dictation_first_flg:
            st.session_state.chain_create_problem = ft.create_chain(
                ct.SYSTEM_TEMPLATE_CREATE_PROBLEM
            )
            st.session_state.dictation_first_flg = False
        # チャット入力以外
        if not st.session_state.chat_open_flg:
            with st.spinner("問題文生成中..."):
                (
                    st.session_state.problem,
                    llm_response_audio,
                ) = ft.create_problem_and_play_audio()

            st.session_state.chat_open_flg = True
            st.session_state.dictation_flg = False
            st.rerun()
        # チャット入力時の処理
        else:
            # チャット欄から入力された場合にのみ評価処理が実行されるようにする
            if not st.session_state.dictation_chat_message:
                st.stop()

            # AIメッセージとユーザーメッセージの画面表示
            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(st.session_state.problem)
            with st.chat_message("user", avatar=ct.USER_ICON_PATH):
                st.markdown(st.session_state.dictation_chat_message)

            # LLMが生成した問題文とチャット入力値をメッセージリストに追加
            st.session_state.messages.append(
                {"role": "assistant", "content": st.session_state.problem}
            )
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": st.session_state.dictation_chat_message,
                }
            )

            # ★ 評価用 LLM を温度0で使う
            with st.spinner("評価結果の生成中..."):
                system_template = ct.SYSTEM_TEMPLATE_EVALUATION.format(
                    llm_text=st.session_state.problem,
                    user_text=st.session_state.dictation_chat_message,
                )
                llm_response_evaluation = ft.create_evaluation(system_template)

            # 評価結果のメッセージリストへの追加と表示
            with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
                st.markdown(llm_response_evaluation)
            st.session_state.messages.append(
                {"role": "assistant", "content": llm_response_evaluation}
            )
            st.session_state.messages.append({"role": "other"})

            # 各種フラグの更新
            st.session_state.dictation_flg = True
            st.session_state.dictation_chat_message = ""
            st.session_state.dictation_count += 1
            st.session_state.chat_open_flg = False

            st.rerun()

    # モード：「日常英会話」
    if st.session_state.mode == ct.MODE_1:
        # 音声入力を受け取って音声ファイルを作成
        audio_input_file_path = (
            f"{ct.AUDIO_INPUT_DIR}/audio_input_{int(time.time())}.wav"
        )
        ft.record_audio(audio_input_file_path)

        # 音声入力ファイルから文字起こしテキストを取得
        with st.spinner("音声入力をテキストに変換中..."):
            transcript = ft.transcribe_audio(audio_input_file_path)
            audio_input_text = transcript.text

        # 音声入力テキストの画面表示
        with st.chat_message("user", avatar=ct.USER_ICON_PATH):
            st.markdown(audio_input_text)

        with st.spinner("回答の音声読み上げ準備中..."):
            # ユーザー入力値をLLMに渡して回答取得
            llm_response = st.session_state.chain_basic_conversation.predict(
                input=audio_input_text
            )

            # LLMからの回答を音声データに変換
            llm_response_audio = st.session_state.openai_obj.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=llm_response,
            )

            # 一旦mp3形式で音声ファイル作成後、wav形式に変換
            audio_output_file_path = (
                f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
            )
            ft.save_to_wav(llm_response_audio.content, audio_output_file_path)

        # 音声ファイルの読み上げ
        ft.play_wav(audio_output_file_path, speed=st.session_state.speed)

        # AIメッセージの画面表示とリストへの追加
        with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
            st.markdown(llm_response)

        # ユーザー入力値とLLMからの回答をメッセージ一覧に追加
        st.session_state.messages.append(
            {"role": "user", "content": audio_input_text}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": llm_response}
        )

    # モード：「シャドーイング」
    # 「シャドーイング」ボタン押下時か、「英会話開始」ボタン押下時
    if (
        st.session_state.mode == ct.MODE_2
        and (
            st.session_state.shadowing_button_flg
            or st.session_state.shadowing_count == 0
            or st.session_state.shadowing_audio_input_flg
        )
    ):
        if st.session_state.shadowing_first_flg:
            st.session_state.chain_create_problem = ft.create_chain(
                ct.SYSTEM_TEMPLATE_CREATE_PROBLEM
            )
            st.session_state.shadowing_first_flg = False

        if not st.session_state.shadowing_audio_input_flg:
            with st.spinner("問題文生成中..."):
                (
                    st.session_state.problem,
                    llm_response_audio,
                ) = ft.create_problem_and_play_audio()

        # 音声入力を受け取って音声ファイルを作成
        st.session_state.shadowing_audio_input_flg = True
        audio_input_file_path = (
            f"{ct.AUDIO_INPUT_DIR}/audio_input_{int(time.time())}.wav"
        )
        ft.record_audio(audio_input_file_path)
        st.session_state.shadowing_audio_input_flg = False

        with st.spinner("音声入力をテキストに変換中..."):
            # 音声入力ファイルから文字起こしテキストを取得
            transcript = ft.transcribe_audio(audio_input_file_path)
            audio_input_text = transcript.text

        # AIメッセージとユーザーメッセージの画面表示
        with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
            st.markdown(st.session_state.problem)
        with st.chat_message("user", avatar=ct.USER_ICON_PATH):
            st.markdown(audio_input_text)

        # LLMが生成した問題文と音声入力値をメッセージリストに追加
        st.session_state.messages.append(
            {"role": "assistant", "content": st.session_state.problem}
        )
        st.session_state.messages.append(
            {"role": "user", "content": audio_input_text}
        )

        # ★ シャドーイングも評価専用 LLM で採点
        with st.spinner("評価結果の生成中..."):
            system_template = ct.SYSTEM_TEMPLATE_EVALUATION.format(
                llm_text=st.session_state.problem,
                user_text=audio_input_text,
            )
            llm_response_evaluation = ft.create_evaluation(system_template)

        # 評価結果のメッセージリストへの追加と表示
        with st.chat_message("assistant", avatar=ct.AI_ICON_PATH):
            st.markdown(llm_response_evaluation)
        st.session_state.messages.append(
            {"role": "assistant", "content": llm_response_evaluation}
        )
        st.session_state.messages.append({"role": "other"})

        # 各種フラグの更新
        st.session_state.shadowing_flg = True
        st.session_state.shadowing_count += 1

        # 「シャドーイング」ボタンを表示するために再描画
        st.rerun()
