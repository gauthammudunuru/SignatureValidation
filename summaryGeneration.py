summary_input = {
    "total_required": len(all_fields),
    "total_signed": signed_count,
    "mismatches": mismatch_list,
    "missing_fields": missing_list,
    "overall_confidence": overall_conf
}

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": str(summary_input)}
    ]
)

print(response.choices[0].message.content)
