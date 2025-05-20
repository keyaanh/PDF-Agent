css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/149/149159.png" alt="Bot Avatar (Bird)" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;" onerror="this.src='https://via.placeholder.com/78x78?text=Bot';">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://scontent-phx1-1.xx.fbcdn.net/v/t39.30808-6/311515817_564349302363659_8837566106528722886_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=6ee11a&_nc_ohc=cYJEAriLr_sQ7kNvwG0vIaD&_nc_oc=AdnYLvDsnV0NWHVirQgAADFmPvJbdiaVRtEFYPDzmin9C0fIzp_XdIfRRSZIJsAp-oOZ5xbJ6ITZaqsAkw6CeuwU&_nc_zt=23&_nc_ht=scontent-phx1-1.xx&_nc_gid=PftaHcSmp16-Q42he9hVIQ&oh=00_AfLxPtq0UHwTrPVZGKbJTbAz4pfDQLBxu5Z51e99PttPMw&oe=6831CCA6" alt="User Avatar" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;" onerror="this.src='https://via.placeholder.com/78x78?text=User';">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''