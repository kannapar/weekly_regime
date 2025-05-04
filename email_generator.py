import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from content_generator import generate_email_content

# Email account credentials
sender_email = "sender@exampl.com"
password = "" #sender password
receiver_email = "receiver@exampl.com"

# Compose the email
msg = MIMEMultipart("alternative")
msg["From"] = sender_email
msg["To"] = receiver_email
msg["Subject"] = "Regime Auto Report"

body = generate_email_content()
msg.attach(MIMEText(body, "html"))

# Send email
with smtplib.SMTP_SSL("smtp.exampl.com", 465) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    print('Email Sent!')