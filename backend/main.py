from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import logging
from typing import Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class HousePurchaseRequest(BaseModel):
    income: float
    total_monthly_debt: float
    total_liquid_assets: float
    zip_code: str
    credit_score: int
    user_input: str = ""
    model: Literal["gemma3:27b"] = "gemma3:27b"
    

async def generate_ollama_stream(messages: list, model: str):
    """Generator function to stream response chunks from Ollama."""
    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True
    )
    for chunk in stream:
        # Yield each content chunk as a string
        yield chunk['message']['content']


@app.post("/plan-home-purchase")
def generate_house_purchase_plan(request: HousePurchaseRequest):
    try:
        prompt = f"""
        You are a financial advisor specializing in home purchases. Provide a detailed assessment of whether a potential home buyer can afford a home, and offer personalized recommendations.

        Here are the buyer's details:
        - Income: ${request.income} per year
        - Total Monthly Debt: ${request.total_monthly_debt}
        - Total Liquid Assets: ${request.total_liquid_assets}
        - Zip Code: {request.zip_code} (for local property tax estimation)
        - Credit Score: ${request.credit_score} (for interest rate, in addition to current trends)
        - User input: ${request.user_input} (if applicable)

        Based on this information, please provide the following:

        1. **Affordability Assessment:**  Determine if the buyer can realistically afford a home and come up with a price range, considering their income, debts, and assets. Explain your reasoning.
        2. **Estimated Monthly Mortgage Payment:** Calculate the estimated monthly mortgage payment (principal and interest) for a home in their price range, using the provided interest rate.
        3. **Estimated Property Taxes:** Provide an estimate of annual property taxes for the given zip code. (Use a reasonable estimate if exact data isn't available.)
        4. **Estimated Homeowners Insurance:** Provide an estimate for annual homeowners insurance.
        5. **Total Estimated Monthly Housing Costs:** Calculate the total estimated monthly housing costs (mortgage, property taxes, insurance).
        6. **Recommendations:** Offer personalized recommendations to the buyer, such as:
            - Whether they should adjust their desired home price range.
            - Strategies for improving their financial situation (e.g., reducing debt, increasing savings).
            - Advice on securing a mortgage.

        Please provide a clear and concise assessment, using a professional tone, but limit wording as much as possible so users can get a quick response.
        """
        messages = [
            {
                'role': 'user',
                'content': prompt,
            },
        ]

        return StreamingResponse(
            generate_ollama_stream(messages, request.model),
            media_type="text/plain"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)