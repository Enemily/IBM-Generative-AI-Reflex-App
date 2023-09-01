# Common styles for questions and answers.
shadow = "rgb(75, 107, 175) 0px 2px 8px"
chat_margin = "20%"
font_style = "IBM Plex Sans, sans-serif"
font_size = "40px" 

message_style = dict(
    padding="1em",
    border_radius="5px",
    margin_y="0.5em",
    # box_shadow=shadow,
    max_width="30em",
    display="inline-block",
    font_family=font_style,
    font_size=font_size,
)

dialogue = dict(
    padding="1em",
    border_radius="10px",
    margin_y="0.5em",
    max_width="30em",
    display="inline-block",
    font_style=font_style,
    font_weight="normal",
)

user_message_style = dialogue | dict(
    bg = "#000000" ,
    text_color = "#FFFFFF",
    margin_left=chat_margin,
)

bot_message_style = dialogue | dict (
    # bg = "#d3d3d3",
    text_color = "#000000",
    margin_right=chat_margin,
)

# # Set specific styles for questions and answers.
# user_message_style = message_style2 | dict(
#     bg="#F5EFFE", margin_left=chat_margin
# )
# bot_message_style = message_style2 | dict(
#     bg="#d3d3d3", margin_right=chat_margin
# )

# Styles for the action bar.
input_style = dict(
    border_width="1px", padding="1em", font_style=font_style, border_color = "#000000", font_weight="normal",
)
button_style = dict(
        background_color="#CCCCCC",  
        color="#000000",            
        border_radius="10px",        
        # padding="0.5em 1em",        
        font_family=font_style,
        font_size = "15px",
        cursor="pointer", 
        font_weight="normal",         
)