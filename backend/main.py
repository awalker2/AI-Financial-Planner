from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import logging
from typing import Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class HousePurchaseQuery(BaseModel):
    income: float
    interest_rate: float
    total_monthly_debt: float
    total_liquid_assets: float
    zip_code: str
    credit_score: int
    user_input: str = ""
    model: Literal["gemma3:27b"] = "gemma3:27b"
    

@app.post("/plan-home-purchase")
def generate_house_purchase_plan(query: HousePurchaseQuery):
    try:
        prompt = f"""
        You are a financial advisor specializing in home purchases. Provide a detailed assessment of whether a potential home buyer can afford a home, and offer personalized recommendations.

        Here are the buyer's details:
        - Income: ${query.income} per year
        - Interest Rate: {query.interest_rate}%
        - Total Monthly Debt: ${query.total_monthly_debt}
        - Total Liquid Assets: ${query.total_liquid_assets}
        - Zip Code: {query.zip_code} (for local property tax estimation)
        - User input: ${query.user_input} (if applicable)

        Based on this information, please provide the following:

        1. **Affordability Assessment:**  Determine if the buyer can realistically afford a home in the desired price range, considering their income, debts, and assets. Explain your reasoning.
        2. **Estimated Monthly Mortgage Payment:** Calculate the estimated monthly mortgage payment (principal and interest) for a home in the desired price range, using the provided interest rate.
        3. **Estimated Property Taxes:** Provide an estimate of annual property taxes for the given zip code. (Use a reasonable estimate if exact data isn't available.)
        4. **Estimated Homeowners Insurance:** Provide an estimate for annual homeowners insurance.
        5. **Total Estimated Monthly Housing Costs:** Calculate the total estimated monthly housing costs (mortgage, property taxes, insurance).
        6. **Recommendations:** Offer personalized recommendations to the buyer, such as:
            - Whether they should adjust their desired home price range.
            - Strategies for improving their financial situation (e.g., reducing debt, increasing savings).
            - Advice on securing a mortgage.

        Please provide a clear and concise assessment, using a professional tone.
        """

        response = ollama.chat(
            model=query.model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
        )
        return {"plan": response['message']['content']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)