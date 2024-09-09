# str8.py
#   Program to count time from a certain event

try:
    from Tkinter import * 
except ImportError:
    from tkinter import *  
from datetime import *
from threading import *

root = Tk()
root.title("STR8")
root.resizable(width=False, height=False)

def main():
    print_it()


def stop():
    root.destroy()


def display():
    event, today, str8, seconds, minutes, hours, days, weeks, years = calc()
    Label(root,
          text="You have been STR8 for:\n",
          font="Verdana 8 bold").grid(row=0, sticky=W)
    Label(root,
          text="Years: "
               + str(round(years, 2)),
          font="Verdana 8").grid(row=1, sticky=W)
    Label(root,
          text="Weeks: "
               + str(round(weeks, 2)),
          font="Verdana 8").grid(row=2, sticky=W)
    Label(root,
          text="Days: "
               + str(round(days, 2)),
          font="Verdana 8").grid(row=3, sticky=W)
    Label(root,
          text="Hours: "
               + str(round(hours, 2)),
          font="Verdana 8").grid(row=4, sticky=W)
    Label(root,
          text="Minutes: "
               + str(round(minutes, 2)),
          font="Verdana 8").grid(row=5, sticky=W)
    Label(root,
          text="Seconds: "
               + str(round(str8.total_seconds())),
          font="Verdana 8").grid(row=6, sticky=W)
    Button(root,
           text="EXIT",
           font="Verdana 8",
           height=1,
           width=19,
           command=stop).grid(row=7)


def calc():
    event = datetime(2017, 3, 29, 13, 45, 0)
    today = datetime.now()

    str8 = today - event

    seconds = str8.total_seconds()
    minutes = str8.total_seconds() / 60
    hours = minutes / 60
    days = hours / 24
    weeks = days / 7
    years = weeks / 52

    return event, today, str8, seconds, minutes, hours, days, weeks, years


def print_it():
    t = Timer(1.0, print_it)
    calc()
    try:
        display()
    except RuntimeError:
        pass
    else:
        t.start()


main()
root.mainloop()





# from Tkinter import *
# import tkFont
# import os
# import glob
# import time
# import threading
# import Image 
# import Queue


# def update_temp(queue):
#     """ Read the temp data. This runs in a background thread. """
#     while True:
#         #   28-000005c6ba08
#         i = "28-000005c6ba08"
#         base_dir = '/sys/bus/w1/devices/'
#         device_folder = glob.glob(base_dir + i)[0]
#         device_file = device_folder + '/w1_slave'

#         tempread=round(read_temp(),1)

#         # Pass the temp back to the main thread.
#         queue.put(tempread)
#         time.sleep(5)

# class Gui(object):
#     def __init__(self, queue):
#         self.queue = queue

#         #Make the window
#         self.root = Tk() 
#         self.root.wm_title("Home Management System")
#         self.root.minsize(1440,1000)

#         self.equipTemp = StringVar()   
#         self.equipTemp1 = StringVar()
#         self.equipTemp2 = StringVar()       

#         self.customFont = tkFont.Font(family="Helvetica", size=16)

#         #   1st floor Image
#         img = Image.open("HOUSE-PLANS-01.png") 
#         photo = ImageTk.PhotoImage(img)

#         Label1=Label(self.root, image=photo)
#         Label1.place(x=100, y=100)

#         #   2nd floor
#         img2 = Image.open("HOUSE-PLANS-02.png")
#         photo2 = ImageTk.PhotoImage(img2)

#         Label1=Label(self.root, image=photo2)
#         Label1.place(x=600, y=100)

#         #   Basement image
#         img3 = Image.open("HOUSE-PLANS-03.png")
#         photo3 = ImageTk.PhotoImage(img3)

#         Label1=Label(self.root, image=photo3)
#         Label1.place(x=100, y=500)

#         #   Attic Image
#         img4 = Image.open("HOUSE-PLANS-04.png")
#         photo4 = ImageTk.PhotoImage(img4)

#         Label1=Label(self.root, image=photo4)
#         Label1.place(x=600, y=500)

#         #   House Isometric Image
#         img5 = Image.open("house-iso.png")
#         photo5 = ImageTk.PhotoImage(img5)

#         Label1=Label(self.root, image=photo5)
#         Label1.place(x=1080, y=130)

#         #Garage Temp Label
#         Label2=Label(self.root, textvariable=self.equipTemp, width=6, justify=RIGHT, font=self.customFont)
#         Label2.place(x=315, y=265)

#         print "start monitoring and updating the GUI"

#         # Schedule read_queue to run in the main thread in one second.
#         self.root.after(1000, self.read_queue)

#     def read_queue(self):
#         """ Check for updated temp data"""
#         try:
#             temp = self.queue.get_nowait()
#             self.equipTemp.set(temp)
#         except Queue.Empty:
#             # It's ok if there's no data to read.
#             # We'll just check again later.
#             pass
#         # Schedule read_queue again in one second.
#         self.root.after(1000, self.read_queue)

# if __name__ == "__main__":
#     queue = Queue.Queue()
#     # Start background thread to get temp data
#     t = threading.Thread(target=update_temp, args=(queue,))
#     t.start()
#     print "starting app"
#     # Build GUI object
#     gui = Gui(queue)
#     # Start mainloop
#     gui.root.mainloop()
