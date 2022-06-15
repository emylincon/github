import streamlit as st
import github

con_obj = github.Contributions()

st.title("Github Analysis")
st.write("This app gives a dashboard of your github contributions")

gitname = st.sidebar.text_input(
    label="Git Repository", placeholder="Enter github repo")

data = con_obj.get_query(gitname)

stat_obj = github.Statistics(data=data)

month_stat = stat_obj.month_contributions()

# st.write(month_stat)
st.write("# Monthly Contributions Bar Chart")
st.bar_chart(month_stat)
