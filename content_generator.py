from regime_module import RegimeIdentification
import warnings
warnings.filterwarnings('ignore')

def regime_output():
    regime = RegimeIdentification().get_sp500_regime()
    if regime.iloc[-1] == 1:
        return "Normal"
    elif regime.iloc[-1] == -1:
        return 'Crash'

def generate_email_content():

    body = f"""
    <html>
      <body>
        <p>This Week is {regime_output()} </p>
        <p> Happy Investing!</p>
      </body>
    </html>
    """
    return body