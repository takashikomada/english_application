import streamlit as st
import os
import time
from pathlib import Path
import wave
import pyaudio
from pydub import AudioSegment
from audiorecorder import audiorecorder
import numpy as np
from scipy.io.wavfile import write
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import constants as ct


def record_audio(audio_input_file_path):
    """
    音声入力を受け取って音声ファイルを作成
    """

    audio = audiorecorder(
        start_prompt="発話開始",
        pause_prompt="やり直す",
        stop_prompt="発話終了",
        start_style={"color": "white", "background-color": "black"},
        pause_style={"color": "gray", "background-color": "white"},
        stop_style={"color": "white", "background-color": "black"},
    )

    if len(audio) > 0:
        audio.export(audio_input_file_path, format="wav")
    else:
        st.stop()


def transcribe_audio(audio_input_file_path):
    """
    音声入力ファイルから文字起こしテキストを取得
    Args:
        audio_input_file_path: 音声入力ファイルのパス
    """

    with open(audio_input_file_path, "rb") as audio_input_file:
        transcript = st.session_state.openai_obj.audio.transcriptions.create(
            model="whisper-1",
            file=audio_input_file,
            language="en",
        )

    # 音声入力ファイルを削除
    os.remove(audio_input_file_path)

    return transcript


def save_to_wav(llm_response_audio, audio_output_file_path):
    """
    一旦mp3形式で音声ファイル作成後、wav形式に変換
    Args:
        llm_response_audio: LLMからの回答の音声データ
        audio_output_file_path: 出力先のファイルパス
    """

    temp_audio_output_filename = (
        f"{ct.AUDIO_OUTPUT_DIR}/temp_audio_output_{int(time.time())}.mp3"
    )
    with open(temp_audio_output_filename, "wb") as temp_audio_output_file:
        temp_audio_output_file.write(llm_response_audio)

    audio_mp3 = AudioSegment.from_file(temp_audio_output_filename, format="mp3")
    audio_mp3.export(audio_output_file_path, format="wav")

    # 音声出力用に一時的に作ったmp3ファイルを削除
    os.remove(temp_audio_output_filename)


def play_wav(audio_output_file_path, speed=1.0):
    """
    音声ファイルの読み上げ
    Args:
        audio_output_file_path: 音声ファイルのパス
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
    """

    # 音声ファイルの読み込み
    audio = AudioSegment.from_wav(audio_output_file_path)

    # 速度を変更（pydub で波形を変形）
    if speed != 1.0:
        modified_audio = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * speed)},
        )
        # 元の frame_rate に戻してピッチを維持
        modified_audio = modified_audio.set_frame_rate(audio.frame_rate)
        modified_audio.export(audio_output_file_path, format="wav")

    # ブラウザ側で再生するためにバイト列に変換
    with open(audio_output_file_path, "rb") as f:
        audio_bytes = f.read()

    # 🔸 ディクテーション用にセッションに保存（rerun 対策）
    st.session_state["dictation_audio_bytes"] = audio_bytes

    # この実行でもプレイヤーを表示
    st.audio(audio_bytes, format="audio/wav")

    # ※ Cloud ではファイル削除はせず、サーバ側のクリーンアップに任せる
    # os.remove(audio_output_file_path)


def create_chain(system_template):
    """
    LLMによる回答生成用のChain作成
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    chain = ConversationChain(
        llm=st.session_state.llm,
        memory=st.session_state.memory,
        prompt=prompt,
    )

    return chain


def create_problem_and_play_audio():
    """
    問題生成と音声ファイルの再生
    """

    # 問題文を生成するChainを実行し、問題文を取得
    problem = st.session_state.chain_create_problem.predict(input="")

    # LLMからの回答を音声データに変換
    llm_response_audio = st.session_state.openai_obj.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=problem,
    )

    # 音声ファイルの作成
    audio_output_file_path = (
        f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
    )
    save_to_wav(llm_response_audio.content, audio_output_file_path)

    # 音声ファイルの読み上げ
    play_wav(audio_output_file_path, st.session_state.speed)

    return problem, llm_response_audio


def create_evaluation(system_template: str) -> str:
    """
    評価専用LLMを使って、問題文とユーザー回答を評価する
    Args:
        system_template: constants.SYSTEM_TEMPLATE_EVALUATION を format した文字列
    """

    messages = [
        SystemMessage(content=system_template),
        HumanMessage(
            content="上記の条件にしたがって、問題文とユーザー回答を評価してください。"
        ),
    ]

    response = st.session_state.eval_llm.invoke(messages)
    return response.content
