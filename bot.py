from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import BOT_TOKEN
from uniqulize import (
    IMAGE_INPUT_SUFFIXES,
    VIDEO_INPUT_SUFFIXES,
    UniqulizeOptions,
    document_suffix_from_name,
    process_bytes_stateless,
    VIDEO_DYNAMICS_TEMPORARILY_DISABLED,
)

MEDIA_MODE_IMAGE = "image"
MEDIA_MODE_VIDEO = "video"
CALLBACK_MEDIA_PREFIX = "media:"
CALLBACK_MEDIA_BACK = f"{CALLBACK_MEDIA_PREFIX}back"

IMAGE_EXTS = IMAGE_INPUT_SUFFIXES
VIDEO_EXTS = VIDEO_INPUT_SUFFIXES

logger = logging.getLogger(__name__)


def media_mode_keyboard(*, include_back: bool = True) -> InlineKeyboardMarkup:
    video_label = "🎬 Динамика (видео)"
    if VIDEO_DYNAMICS_TEMPORARILY_DISABLED:
        video_label = "🎬 Динамика (видео) — временно недоступно"
    rows = [
        [
            InlineKeyboardButton("🖼 Статика (картинки)", callback_data=f"{CALLBACK_MEDIA_PREFIX}{MEDIA_MODE_IMAGE}"),
            InlineKeyboardButton(video_label, callback_data=f"{CALLBACK_MEDIA_PREFIX}{MEDIA_MODE_VIDEO}"),
        ]
    ]
    if include_back:
        rows.append([InlineKeyboardButton("◀️ Назад", callback_data=CALLBACK_MEDIA_BACK)])
    return InlineKeyboardMarkup(rows)


def _get_media_mode(ctx: ContextTypes.DEFAULT_TYPE) -> str | None:
    raw = (ctx.user_data or {}).get("media_mode")
    if raw in (MEDIA_MODE_IMAGE, MEDIA_MODE_VIDEO):
        return raw
    return None


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    video_note = ""
    if VIDEO_DYNAMICS_TEMPORARILY_DISABLED:
        video_note = "\n\n⚠️ <b>Динамика (видео) временно недоступна</b> — сейчас работает только статика (картинки)."
    text = (
        "👋 Привет! Я помогаю уникализировать креативы: убираю метаданные и собираю новый файл "
        "(для картинок — с микро-изменениями геометрии/цвета/шума: визуально почти как оригинал, но файл и отпечаток другие).\n\n"
        "🖼 <b>Картинки: JPG, JPEG, PNG, WEBP, BMP, GIF (первый кадр), TIFF, HEIC, HEIF, AVIF.</b>\n"
        "🎥 <b>Видео: MP4, MOV, M4V, WEBM, MKV, AVI, MPEG, MPG, M2TS, TS, FLV, WMV, 3GP, OGV.</b>\n\n"
        "📎 <b>Лучше отправлять как документ, чтобы Telegram не пережимал качество.</b>\n\n"
        "👇 Выбери режим — от него зависит, что я обрабатываю.\n\n"
        "ℹ️ /help — как пользоваться.\n"
        "ℹ️ /info — что именно чистится и пересобирается (для FB)."
        + video_note
    )
    await update.message.reply_text(
        text,
        reply_markup=media_mode_keyboard(),
        parse_mode=ParseMode.HTML,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "📖 Как пользоваться\n"
        "1️⃣ /start → выбери «Статика» (только картинки) или «Динамика» (только видео).\n"
        "2️⃣ Пришли файл (удобнее как документ).\n\n"
        "🖼 Картинки: jpg, jpeg, png, webp, bmp, gif, tif, tiff, heic, heif, avif.\n"
        "🎥 Видео: mp4, mov, m4v, webm, mkv, avi, mpeg, mpg, m2ts, ts, flv, wmv, 3gp, ogv.\n\n"
        "⏳ Обработка может занять время: для изображений пересобирается JPEG без EXIF/ICC.\n\n"
        "⚙️ Важно\n"
        "• Для видео используется ffmpeg (бот попробует скачать локально автоматически).\n"
        "• HEIC/AVIF — пакеты из requirements.txt.\n\n"
        "⚠️ Про модерацию FB: отказ зависит не только от файла.\n\n"
        "📋 /info — полный список того, что бот снимает с файла и что добавляет заново."
    )
    await update.message.reply_text(text)


async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Подробное описание очистки и уникализации для пользователя (FB / доверие к файлу)."""
    if not update.message:
        return

    intro = (
        "ℹ️ <b>Что я умею</b>\n\n"
        "Убираю с крео старые метаданные и следы редактора/экспорта, пересобираю картинку или видео "
        "и подставляю параметры, похожие на обычный мобильный контент — так файл выглядит "
        "аккуратнее и «трастовее» для площадок вроде Facebook.\n\n"
        "<b>Что именно чищу и чем заменяю</b>\n\n"
        "<b>🖼 Картинки (статика)</b>\n"
        "• <b>Снимаю с исходника:</b> исходный EXIF, GPS, XMP, IPTC, вложенные данные редактора; "
        "цветовые профили из чужого файла не переношу — кадр пересобирается в чистый RGB.\n"
        "• <b>Перекодирую JPEG заново</b> (новые таблицы квантования и битстрим): визуально "
        "режим FB старается оставить кадр как оригинал, но байты файла — другие.\n"
        "• <b>Микро-уникализация пикселей:</b> незаметный шум на весь кадр и сдвиг одного пикселя — "
        "чтобы хеш и отпечаток файла не совпадали с исходником.\n"
        "• <b>Ресайз/геометрия</b> (если включена не «идентичная» обработка): случайный алгоритм "
        "масштабирования (LANCZOS / BILINEAR / BICUBIC) — ещё один отличительный след.\n"
        "• <b>После сохранения вшиваю синтетический EXIF</b> под популярные камеры (iPhone / Samsung): "
        "даты «недавно», модель, серийники, экспозиция — <b>без GPS</b>.\n"
        "• <b>Миниатюра (thumbnail) в EXIF</b> — как у нормального телефонного фото, а не «пустой» снимок.\n"
        "• <b>ICC-профиль:</b> чужие профили убираются; в финале подставляется стандартный "
        "<b>sRGB (IEC 61966-2.1)</b> — типичная картина для мобильных снимков.\n"
    )

    part2 = (
        "<b>🎥 Видео (динамика)</b>\n"
        "• <b>Обнуляю контейнерные метаданные:</b> без старых тегов, глав и обложек — "
        "подаётся новый MP4 с <code>-map_metadata -1</code>, без субтитров и лишних дорожек.\n"
        "• <b>Режим совместимости с FB:</b> перекодирование в H.264 (libx264), YUV 4:2:0, "
        "случайные параметры GOP/качества — новый видеопоток.\n"
        "• <b>Очистка служебных NAL в H.264:</b> битстрим-фильтр <code>h264_metadata=delete_filler=1</code> "
        "(убирает лишние filler NAL).\n"
        "• <b>Меньше «следов софта» в заголовках:</b> флаг <code>-bitexact</code> у ffmpeg "
        "(менее болтливый/типовой контейнерный след).\n"
        "• <b>Картинка:</b> лёгкая цветокоррекция, шум, иногда отражение; сверху — "
        "<b>почти невидимый слой шума через overlay</b> (~0.5% смеси), чтобы кадр отличался от исходника.\n"
        "• <b>Частота кадров:</b> лёгкая подстройка через фильтр <code>fps=…</code> "
        "или выходной <code>-r</code> (например 24 / 25 / 29.97 / 30) — микро-отличие по таймингам.\n"
        "• <b>Аудио:</b> принудительно <b>44100 Hz</b>; лёгкий <code>aecho</code> "
        "или инверсия полярности стерео через <code>pan</code> — дорожка новая по сигналу; "
        "громкость и темп слегка меняются в разумных пределах.\n\n"
        "<b>🔒 Как я веду себя в Telegram</b>\n"
        "• <b>Zero storage:</b> я не храню историю файлов/хэшей и не веду базу обработок.\n"
        "• <b>Stateless:</b> файл обрабатывается как поток (в памяти/временной среде) и сразу отдаётся пользователю.\n\n"
        "<i>Итог: на выходе новый файл — без старого «паспорта» и с обновлённым битстримом под мобильный вид.</i>"
    )

    await update.message.reply_text(intro, parse_mode=ParseMode.HTML)
    await update.message.reply_text(part2, parse_mode=ParseMode.HTML)


async def on_media_mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    if not q:
        return
    data = (q.data or "").strip()

    if data == CALLBACK_MEDIA_BACK:
        context.user_data.pop("media_mode", None)
        await q.answer("↩️ Сброшено")
        msg = q.message
        if not msg:
            return
        try:
            if msg.text and msg.text.startswith("👋"):
                await msg.edit_reply_markup(reply_markup=media_mode_keyboard())
                await msg.reply_text("↩️ Ок, режим сброшен. 👇 Выбери снова.")
            else:
                await msg.edit_text(
                    "↩️ Режим сброшен. Выбери снова 👇",
                    reply_markup=media_mode_keyboard(),
                )
        except Exception:
            await msg.reply_text(
                "↩️ Режим сброшен. Выбери снова 👇",
                reply_markup=media_mode_keyboard(),
            )
        return

    m = re.match(rf"^{re.escape(CALLBACK_MEDIA_PREFIX)}(image|video)$", data)
    if not m:
        await q.answer()
        return
    mode = MEDIA_MODE_IMAGE if m.group(1) == "image" else MEDIA_MODE_VIDEO
    if mode == MEDIA_MODE_VIDEO and VIDEO_DYNAMICS_TEMPORARILY_DISABLED:
        await q.answer("⏳ Динамика (видео) временно недоступна. Выбери «Статика (картинки)».", show_alert=True)
        return
    context.user_data["media_mode"] = mode
    label = "🖼 Статика (изображения)" if mode == MEDIA_MODE_IMAGE else "🎬 Динамика (видео)"
    await q.answer(f"✅ {label}")
    try:
        await q.edit_message_reply_markup(reply_markup=media_mode_keyboard())
    except Exception:
        pass
    await q.message.reply_text(
        f"✅ Выбрано: {label}\n"
        "📎 Можно присылать файл.\n"
        "↩️ «Назад» ниже — сбросить выбор режима.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("◀️ Назад", callback_data=CALLBACK_MEDIA_BACK)]]),
    )


def _classify_incoming(
    doc,
    vid,
    photo,
) -> tuple[str | None, str, str | None]:
    """Returns (kind, suffix, err). kind is image|video, or None if unknown/bad."""
    if doc:
        suffix = document_suffix_from_name(doc.file_name, ".bin")
        if suffix == ".bin":
            return None, suffix, "🤔 Не понял формат. Список расширений — в /start и /help (лучше как документ 📎)."
        if suffix in IMAGE_EXTS:
            return MEDIA_MODE_IMAGE, suffix, None
        if suffix in VIDEO_EXTS:
            return MEDIA_MODE_VIDEO, suffix, None
        return None, suffix, "🤔 Не понял формат. Список расширений — в /start и /help (лучше как документ 📎)."
    if vid:
        return MEDIA_MODE_VIDEO, ".mp4", None
    if photo:
        return MEDIA_MODE_IMAGE, ".jpg", None
    return None, ".bin", None


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    doc = update.message.document
    vid = update.message.video
    photo = update.message.photo[-1] if update.message.photo else None

    if not any([doc, vid, photo]):
        return

    kind, suffix, classify_err = _classify_incoming(doc, vid, photo)
    if classify_err:
        await update.message.reply_text(classify_err)
        return

    assert kind is not None

    mode = _get_media_mode(context)
    if mode is None:
        await update.message.reply_text(
            "⏳ Режим не выбран.\n\n"
            "Сначала нажми «Статика» или «Динамика» (на /start или кнопками ниже) — "
            "иначе уникализация не запускается.",
            reply_markup=media_mode_keyboard(),
        )
        return

    if kind != mode:
        if mode == MEDIA_MODE_IMAGE:
            await update.message.reply_text(
                "🖼 Сейчас включена статика (только изображения).\n"
                "🎬 Для видео нажми «◀️ Назад» или /start и переключись на «Динамика».",
                reply_markup=media_mode_keyboard(),
            )
        else:
            await update.message.reply_text(
                "🎬 Сейчас включена динамика (только видео).\n"
                "🖼 Для картинок нажми «◀️ Назад» или /start и переключись на «Статику».",
                reply_markup=media_mode_keyboard(),
            )
        return

    opts = UniqulizeOptions(preset="strong", preserve_visual=False, synthetic_exif=True)

    await update.message.chat.send_action(ChatAction.TYPING)

    status_msg = None
    try:
        if doc:
            tg_file = await doc.get_file()
        elif vid:
            tg_file = await vid.get_file()
        else:
            tg_file = await photo.get_file()

        blob = await tg_file.download_as_bytearray()
        data = bytes(blob)

        status_msg = await update.message.reply_text("⏳ Идёт обработка… Это может занять немного времени 😎")
        await update.message.chat.send_action(ChatAction.UPLOAD_DOCUMENT)

        loop = asyncio.get_running_loop()
        payload, out_name = await loop.run_in_executor(
            None, lambda: process_bytes_stateless(data, suffix=suffix, mode="uniqulize", options=opts)
        )

        if status_msg is not None:
            try:
                await status_msg.delete()
            except Exception:
                pass
            status_msg = None

        mode_ru = "статика 🖼" if mode == MEDIA_MODE_IMAGE else "динамика 🎬"
        await update.message.reply_document(
            document=payload,
            filename=out_name,
            caption=f"✅ Готово! Режим: {mode_ru}",
        )
        context.user_data.pop("media_mode", None)
        await update.message.reply_text(
            "🔁 Что уникализируем дальше?\n\n"
            "👇 Сначала снова нажми «Статика» или «Динамика» — без этого следующий файл я не обрабатываю.",
            reply_markup=media_mode_keyboard(),
        )
        logger.info(
            "Успешная уникализация: chat_id=%s mode=%s file=%s",
            update.effective_chat.id if update.effective_chat else "?",
            mode,
            out_name,
        )
    except Exception as e:
        if status_msg is not None:
            try:
                await status_msg.delete()
            except Exception:
                pass
        logger.exception("Ошибка уникализации: %s", e)
        await update.message.reply_text(f"😓 Ошибка: {e}")


async def handle_wrong_without_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сообщения без медиа, не команды — если режим не выбран, подсказка и клавиатура (group 1)."""
    msg = update.effective_message
    if not msg:
        return
    if msg.document or msg.video or msg.photo:
        return
    text = (msg.text or "").strip()
    if text.startswith("/"):
        return
    if _get_media_mode(context) is not None:
        return
    await msg.reply_text(
        "⚠️ Неверное действие.\n\n"
        "Сначала нажми «Статика» или «Динамика» — без выбора режима уникализация не запускается.\n\n"
        "👇 Выбери режим:",
        reply_markup=media_mode_keyboard(),
    )


async def _post_init(application: Application) -> None:
    me = await application.bot.get_me()
    logger.info("Подключено к Telegram: @%s (id=%s)", me.username or "—", me.id)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    token = (BOT_TOKEN or "").strip()
    if not token or token == "PASTE_YOUR_TOKEN_HERE":
        raise RuntimeError("Открой config.py и вставь BOT_TOKEN.")

    logger.info("Старт Uniqulizer: сборка приложения…")

    app = Application.builder().token(token).post_init(_post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("info", cmd_info))
    app.add_handler(CallbackQueryHandler(on_media_mode_callback, pattern=r"^media:(image|video|back)$"))

    app.add_handler(
        MessageHandler(filters.Document.ALL | filters.VIDEO | filters.PHOTO, handle_media),
        group=0,
    )
    app.add_handler(MessageHandler(filters.ALL, handle_wrong_without_mode), group=1)

    logger.info("Запуск long polling (Ctrl+C — остановка)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
