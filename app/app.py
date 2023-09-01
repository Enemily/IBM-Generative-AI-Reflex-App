# Import reflex, style and state
import reflex as rx
from app import style
from app.state import State


def qa(question:str, answer:str) -> rx.Component: 
    return rx.box(
        rx.box(rx.text(question, style=style.user_message_style), text_align="right"), 
        rx.box(rx.text(answer, style=style.bot_message_style), text_align="left"),
        margin_y="1em" 
    )
    

def chat() -> rx.Component: 
    return rx.box(
        rx.foreach(State.chat_history, lambda messages: qa(messages[0], messages[1]))
    )

def action_bar() -> rx.Component: 
    return rx.hstack(
        rx.input(placeholder="Ask a question", on_blur=State.set_question, style=style.input_style), 
        rx.button("Ask", on_click=State.existing_chain, style=style.button_style),
    )

def navbar():
    return rx.box(
        rx.hstack(
            rx.text("watsonx.ai + LangChain"),
            style=style.message_style
        ),
        rx.spacer(),
        padding="1em",
        width="100%",
        top="5px",
    )

def index() -> rx.Component:
    """The main view."""
    return rx.container(navbar(), chat(), rx.hstack(
                                            rx.input(
                                                placeholder="Ask a question",
                                                on_blur=State.set_question,
                                                style=style.input_style,
                                                flex="1"  # Allow the input to grow and take available space
                                            ),
                                            rx.button(
                                                "About current file",
                                                on_click=State.answer,
                                                style=style.button_style
                                            ),
                                            rx.responsive_grid(
                                                rx.text(State.img[-1]),
                                                style=style.button_style
                                            )
                                        ),rx.hstack(  
                                            rx.upload(
                                                rx.box(
                                                    rx.button(
                                                        "Select new file",
                                                        style=style.button_style
                                                    ),
                                                    flex_direction="column",
                                                    spacing="0.5em"
                                                ),
                                                multiple=True,
                                                accept={
                                                    "application/pdf": [".pdf"],
                                                    "image/png": [".png"],
                                                    "image/jpeg": [".jpg", ".jpeg"],
                                                    "image/gif": [".gif"],
                                                    "image/webp": [".webp"],
                                                    "text/html": [".html", ".htm"],
                                                },
                                                max_files=5,
                                                disabled=False,
                                                on_keyboard=True,
                                            ),
                                            rx.button(
                                                "Upload",
                                                on_click= lambda:  State.handle_upload(
                                                    rx.upload_files()
                                                ),
                                                style=style.button_style
                                            ),
                                            flex_direction="row",  # Lay out the components horizontally
                                            align_items="center",  # Align items vertically in the center
                                            padding="1em"
                                        ))



# Add state and page to the app.
app = rx.App()
app.add_page(index)
app.compile()



