import random
import os
import warnings
import logging
import streamlit as st
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

st.set_page_config(page_title="Школа 21: RAG Ментор", page_icon="👑", layout="wide")

OBSIDIAN_VAULT = r""path/to/your/obsidian/vault""
CHROMA_DB_PATH = "./chroma_db_vault"
MODEL_NAME = "qwen2.5:7b"

@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if os.path.exists(CHROMA_DB_PATH):
        print("[+] Загрузка существующей векторной базы...")
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    
    print("[*] Глубокое сканирование всего архива Obsidian...")
    documents = []
    
    for root, dirs, files in os.walk(OBSIDIAN_VAULT):
        if any(x in root for x in ["attachments", ".obsidian", "old"]):
            continue
            
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        if len(text.strip()) > 10:
                            from langchain_core.documents import Document
                            # Сохраняем путь к папке строго как СТРОКУ для корректного поиска
                            rel_dir = str(os.path.relpath(root, OBSIDIAN_VAULT))
                            documents.append(Document(page_content=text, metadata={"source": file, "folder": rel_dir}))
                except Exception as e:
                    pass
                    
    if not documents:
        raise ValueError("Критическая ошибка: Не найдено ни одного текстового .md файла.")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
    print(f"[+] УСПЕХ! Проиндексировано {len(chunks)} кусков текста.")
    return vector_db

try:
    db = load_rag_system()
except Exception as e:
    st.error(f"Служба Ollama не отвечает. Убедись, что приложение Ollama запущено. Лог: {e}")
    st.stop()

def get_all_obsidian_folders():
    folder_set = set()
    for root, dirs, files in os.walk(OBSIDIAN_VAULT):
        if any(x in root for x in ["attachments", ".obsidian", "old"]):
            continue
        if any(f.endswith(".md") for f in files):
            rel_dir = str(os.path.relpath(root, OBSIDIAN_VAULT))
            if rel_dir != ".":
                folder_set.add(rel_dir)
    return ["Все папки"] + sorted(list(folder_set))

def get_relevant_context(topic_query, folder_filter="Все папки", k=3):
    if folder_filter != "Все папки":
        docs = db.similarity_search(topic_query, k=k, filter={"folder": str(folder_filter)})
    else:
        docs = db.similarity_search(topic_query, k=k)
        
    if not docs or len(docs) == 0:
        return None
    return "\n\n--- СЛЕДУЮЩИЙ КУСОК ЛЕКЦИИ ---\n\n".join([doc.page_content for doc in docs])

def generate_task(topic, folder_filter="Все папки"):
    context = get_relevant_context(topic, folder_filter=folder_filter)
    if context is None:
        return f"⚠️ **Ошибка поиска контента:** Наталия, я не нашел задач по теме **'{topic}'** в выбранной папке."

    prompt = f"""
    Ты — строгий экзаменатор. Твоя задача — извлечь из предоставленного КОНТЕКСТА ЛЕКЦИЙ практическое задание по теме '{topic}' и выдать СТРОГО ЕГО УСЛОВИЕ.
    
    💥 КРИТИЧЕСКОЕ И ЖЕСТКОЕ ОГРАНИЧЕНИЕ (АНТИ-СПОЙЛЕР):
    1. Тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО выводить готовый SQL-код, куски запросов, блоки кода SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY или решения на Python/C++.
    2. Если в тексте лекции написана задача и сразу приведен правильный SQL-запрос (как в примере про student_id, fives, марк = 5) — ты ОБЯЗАН СТЕРЕТЬ И УДАЛИТЬ этот SQL-код! 
    3. Вместо вывода кода сформулируй чистое текстовое задание для студента на основе этого примера. 
       Например, напиши: "Задание: Напишите SQL-запрос, который выведет student_id и количество пятерок...". Но сам код решения не показывай!
    4. Запрещено писать вводные фразы вроде "Этот запрос выглядит правильно...", "Вот условие..." или "Давайте разберем...". Сразу выводи условие задания!
    
    КОНТЕКСТ ИЗ ОБСИДИАНА:
    {context}
    """
    response = ollama.generate(model=MODEL_NAME, prompt=prompt)
    return response['response']


def verify_answer(topic, task_text, student_answer, folder_filter="Все папки"):
    context = get_relevant_context(topic, folder_filter=folder_filter)
    
    prompt = f"""
    Ты — строгий, беспристрастный и точный экзаменатор Школы 21. Твоя задача — проверить ответ студента на основе ГОТОВОГО ОФИЦИАЛЬНОГО РЕШЕНИЯ из лекции.
    
    ЖЕСТКИЕ ПРАВИЛА ПРОВЕРКИ (ЗАПРЕТ НА САМОСТОЯТЕЛЬНЫЕ РАСЧЕТЫ):
    1. Категорически ЗАПРЕЩЕНО решать задачу самостоятельно или выводить свои цифры. Верь ТОЛЬКО тексту лекции ниже.
    2. Внимательно изучи текст лекции. Найди там точные финальные расчеты для этой задачи (включая разделы о математически верном ответе и специфике ввода для Stepik, если они есть).
    3. Сверь ответ студента с данными из лекции. Поставь вердикт ВЕРНО, если студент указал точный математический ответ ИЛИ правильный ответ для ввода в систему тестирования, зафиксированный в лекции. 
    4. Если студент ввел число, которого нет в тексте лекции — ставь вердикт НЕВЕРНО.
    
    КОНТЕКСТ С РЕШЕНИЕМ ИЗ ЛЕКЦИИ OBSIDIAN:
    {context}
    
    ВЫДАННАЯ ЗАДАЧА:
    {task_text}
    
    ОТВЕТ СТУДЕНТА:
    {student_answer}
    
    ТРЕБОВАНИЯ К ФОРМАТУ ТВОЕГО ОТВЕТА:
    1. Поставь четкий вердикт в самом начале: ВЕРНО или НЕВЕРНО.
    2. Выведи блок '### 📘 Сверка с материалами лекции'. Пошагово покажи, совпали ли цифры студента с конспектом.
    3. Оборачивай абсолютно все формулы, дроби и переменные в одиночные знаки доллара `$ formula $`.
    4. Добавь блок '### 📋 Чек-лист усвоения темы' с пустыми квадратами Юникода '⬜', начинающийся с маркеров действия (Знаю, Понимаю, Умею).
    """
    response = ollama.generate(model=MODEL_NAME, prompt=prompt)
    return response['response']


st.sidebar.header("🎯 Панель управления")
mode = st.sidebar.radio("Выбери режим работы:", ["🎓 RAG-Ментор (По Obsidian)", "💬 Свободный чат с Qwen"])

if "task_text" not in st.session_state:
    st.session_state.task_text = ""
if "verification_text" not in st.session_state:
    st.session_state.verification_text = ""
if "active_topic" not in st.session_state:
    st.session_state.active_topic = "Общая тема"

if mode == "💬 Свободный чат с Qwen":
    st.title("💬 Свободный чат с Qwen 2.5")
    st.caption("Режим стандартного ИИ-ассистента (без привязки к твоим лекциям)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if user_msg := st.chat_input("Спроси ИИ о чем угодно..."):
        with st.chat_message("user"):
            st.markdown(user_msg)
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("assistant"):
            res = ollama.generate(model=MODEL_NAME, prompt=user_msg)
            st.markdown(res['response'])
        st.session_state.messages.append({"role": "assistant", "content": res['response']})

else:
    st.title("👑 Персональный ИИ-Ментор")
    st.caption("Локальная RAG-система на базе моих лекций Obsidian")

    st.sidebar.subheader("🎯 Фокус обучения")
    available_folders = get_all_obsidian_folders()
    selected_folder = st.sidebar.selectbox("Выбери папку для поиска:", available_folders)

    st.sidebar.subheader("Настройки сессии")
    target_topic = st.sidebar.text_input("Введи тему для проверки:", placeholder="например: Агрегатные функции")

    col1, col2 = st.sidebar.columns(2)
        
    with col1:
        if st.button("🤖 По теме", use_container_width=True):
            if target_topic:
                with st.spinner("Ищу задачу по теме..."):
                    st.session_state.active_topic = target_topic
                    st.session_state.task_text = generate_task(target_topic, folder_filter=selected_folder)
                    st.session_state.verification_text = ""
            else:
                st.sidebar.warning("Сначала введи тему!")

    with col2:
        if st.button("🎲 Рандом", use_container_width=True):
            with st.spinner("Выбираю случайный файл..."):
                md_files = []
                for root, dirs, files in os.walk(OBSIDIAN_VAULT):
                    if any(x in root for x in ["attachments", ".obsidian", "old"]):
                        continue
                    
                    rel_dir = str(os.path.relpath(root, OBSIDIAN_VAULT))
                    if selected_folder != "Все папки" and rel_dir != selected_folder:
                        continue
                        
                    for file in files:
                        if file.endswith(".md"):
                            md_files.append(os.path.join(root, file))
                
                if md_files:
                    random_file_path = random.choice(md_files)
                    st.session_state.active_topic = os.path.basename(random_file_path).replace(".md", "")
                    
                    with open(random_file_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_content = f.read()
                        
                    prompt = f"""
                    Ты — строгий экзаменатор Школы 21. Перед тобой реальный файл из заметок студента.
                    Найди в этом тексте ОДНУ практическую задачу или пример и выведи СТРОГО её условие.
                    Если готовой задачи нет, сформулируй ОДИН глубокий вопрос по тексту этого файла для проверки знаний.
                    Категорически ЗАПРЕЩЕНО придумывать сюжеты со стороны. Полностью УДАЛИ, СОРТИ И СКРОЙ готовые решения и ответы!
                    
                    ТЕКСТ ФАЙЛА:
                    {file_content}
                    """
                    # ЖЕСТКАЯ ФИКСАЦИЯ: Добавляем отправку запроса в Ollama и сохранение результата
                    res = ollama.generate(model=MODEL_NAME, prompt=prompt)
                    st.session_state.task_text = res['response']
                    st.session_state.verification_text = ""
                else:
                    st.sidebar.error("В этой папке нет .md файлов!")

    if st.session_state.task_text:
        st.subheader(f"📝 Текущее задание по теме: {st.session_state.active_topic}")
        clean_task = st.session_state.task_text.replace(r"\(", "$").replace(r"\)", "$")
        clean_task = clean_task.replace(r"\[", "$$").replace(r"\]", "$$")
        
        # борьба с отображением:  у меня в обсидиан формулы обернуты в $$
        if r"\frac" in clean_task and "$\frac" not in clean_task:
            # Находим саму дробь и аккуратно изолируем её знаками $
            clean_task = clean_task.replace(r"\frac", r"$\frac")
            # Закрываем знак доллара после фигурных скобок дроби
            clean_task = clean_task.replace(r"}", r"}$")
            # Чистим двойные доллары, если они случайно образовались
            clean_task = clean_task.replace("$$", "$")
            
        st.markdown(clean_task)

        
        if "⚠️" not in st.session_state.task_text:
            st.divider()
            st.subheader("📥 Твое решение")
            student_input = st.text_area("Впиши свой ответ / код / рассуждения здесь:", height=150, placeholder="Мой вариант решения...")
            
            if st.button("🚀 Отправить решение на проверку", type="primary"):
                if student_input:
                    with st.spinner("Ментор проверяет решение..."):
                        st.session_state.verification_text = verify_answer(st.session_state.active_topic, st.session_state.task_text, student_input, folder_filter=selected_folder)
                else:
                    st.warning("Нельзя отправить пустой ответ!")


    if st.session_state.verification_text:
        st.divider()
        st.subheader("⚖️ Вердикт и подробный разбор ментора")
        
        raw_text = st.session_state.verification_text
        
        clean_v = raw_text.replace(r"\(", "$").replace(r"\)", "$")
        clean_v = clean_v.replace(r"\[", "$$").replace(r"\]", "$$")
        
        lines = clean_v.split("\n")
        fixed_lines = []
        for line in lines:
            # Если в строке есть математический слэш, но он не обернут в $, оборачиваем всю строку или ее математическую часть
            if r"\\" in line or r"\frac" in line or r"\sim" in line or r"\sum" in line:
                if "$" not in line:
                    # Если строка чисто математическая (содержит спецсимволы), оборачиваем ее целиком в $$
                    line = f"$${line.strip()}$$"
                else:
                    # Если доллары есть, но разметка все равно побита, страхуем замену дробей
                    line = line.replace(r"\frac", r"$\frac").replace(r"}", r"}$").replace("$$", "$")
            fixed_lines.append(line)
            
        final_verification = "\n".join(fixed_lines)
        st.markdown(final_verification)

