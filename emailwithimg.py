import os
import smtplib
import threading

def printit1():
      threading.Timer(10.0, printit).start()
      s = smtplib.SMTP('smtp.gmail.com', 587)
      s.starttls()
      s.login("wcyber23@gmail.com", "ANKITK.AS51")
      TEXT = "Suspected object Detected."
      SUBJECT=" Message from Camera"
      to="priyankadeshmukh3096@gmail.com"
      by="wcyber23@gmail.com"
      message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)
      s.sendmail(by,to, message)
      s.quit()
      print ("mail sent to priyanka ")

def printit():
      threading.Timer(10.0, printit).start()
      from email.mime.text import MIMEText
      from email.mime.image import MIMEImage
      from email.mime.multipart import MIMEMultipart
      fromaddr = 'wcyber23@gmail.com'
      toaddrs  = 'ankitk.as51@gmail.com'
      username = 'wcyber23@gmail.com'
      password = 'ANKITK.AS51'
      server = smtplib.SMTP('smtp.gmail.com:587')
      server.ehlo()
      msg = MIMEMultipart()
      fp = open('ankit.jpg', 'rb')
      img = MIMEImage(fp.read())
      fp.close()
      msg.attach(img)
      server.starttls()
      server.login(username,password)
      server.sendmail(fromaddr, toaddrs, msg.as_string())
      print('sent')
      server.quit()       
      print ("mail sent")

printit()
printit1() 
