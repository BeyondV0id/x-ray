from dotenv import load_dotenv
load_dotenv('.env')
import os
from src.agentic_ai import MedicalAgenticPipeline

pipeline = MedicalAgenticPipeline()
print('Is pipeline active with LangChain + Gemini?', pipeline._chain is not None)

res = pipeline.run_pipeline('Tuberculosis', [[100, 100, 300, 300]], [0.95])
print('\n=== GEMINI LANGCHAIN RESPONSE ===')
print('LLM Powered:', res['llm_powered'])
print('LLM Model:', res['llm_model'])
print('Overall Summary:', res['overall_summary'])
print('\n1. Clinical Summary Narrative:\n', res['clinical_summary'].get('summary'))
print('\n2. Prescribed Medications:')
for m in res['prescription_recommendation'].get('medications', []):
    print(f"   💊 {m.get('drug_name')} ({m.get('dosage')}) - {m.get('frequency')}")
    print(f"      Indication: {m.get('indication')}")
