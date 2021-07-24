#
# WGU C964 Capstone Project
# Equipment Faults in Manufacturing Environments
# Sean Naramor
# July 23, 2021
#
####################################
# This file acts as an entry point into the application as well as a gateway to allow access control to take place
####################################
import streamlit as st
from sessionstate import get
from main import main

session_state = get(password='')

if session_state.password != 'password':
    pwd_placeholder = st.sidebar.empty()
    pwd = pwd_placeholder.text_input("Password:", value="", type="password")
    session_state.password = pwd
    if session_state.password == 'password':
        pwd_placeholder.empty()
        main()
    else:
        st.sidebar.error("the password you entered is incorrect")
else:
    main()
