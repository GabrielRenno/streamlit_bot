import streamlit as st

# Define the list of usernames and passwords
user_data = {
    'ruben.tak@rslt.agency': 'ruben',
    'gabriel.renno@rslt.agency': 'gabriel',
    'nils.jennissen@rslt.agency': 'nils',
    'onassis.nottage@rslt.agency': 'onassis',
    'enriquepeto@yahoo.es': 'xR9p#2',
    'jgonzalez@profe.csm.cat': 'lE0@#1',
    'vgregorio@profe.csm.cat': "Rt5y#1",
    'atomasa@profe.csm.cat': 'Kp2@l9',
    'chidalgo@profe.csm.cat': 'Jf0g98',
    'cmatellan@profe.csm.cat': 'Ls9@t3',
    'scunillera@profe.csm.cat' : 'Gj4l&7',
    'bvega@profe.csm.cat': 'Xk6p@2',
}


def authentication_local():

    def check_password():
        """Returns `True` if the user had a correct password."""

        def login_form():
            """Form with widgets to collect user information"""
            with st.form("Credentials"):
                username = st.text_input("Username", key="username")
                st.text_input("Password", type="password", key="password")
                st.form_submit_button("Log in", on_click=password_entered)

            return username

        def password_entered():
            """Checks whether a password entered by the user is correct."""
            if st.session_state["username"] in user_data and st.session_state["password"] == user_data[st.session_state["username"]]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store the username or password.
                #del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False

        # Return True if the username + password is validated.
        if st.session_state.get("password_correct", False):
            return True

        # Show inputs for username + password.
        username = login_form()
        if "password_correct" in st.session_state:
            st.error("😕 User not known or password incorrect")
        return False


    if not check_password():
        st.stop()

    return