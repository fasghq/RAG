FROM python:3.12-slim

# Отключаем создание файлов .pyc и буферизацию вывода
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Устанавливаем рабочую директорию в контейнере
WORKDIR /home

# Копируем файлы проекта в рабочую директорию
# COPY . .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn python-multipart langchain langchain_community langchain_docling docling faiss-cpu rank_bm25 sentence-transformers openai

# Открываем порт 8000 для доступа к серверу
EXPOSE 8000

# Команда для запуска сервера FastAPI через uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
