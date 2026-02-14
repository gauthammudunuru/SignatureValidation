pdf_path = "sample_docs/sample_contract.pdf"
pages = pdf_to_images(pdf_path)

all_fields = []
field_conf = []
presence_conf = []
identity_conf = []

for page_number, image in pages:
    result = extract_fields(image, page_number)
    
    for field in result["fields"]:
        all_fields.append(field)
        field_conf.append(field["field_detection_confidence"])
        
print("Total required signatures:", len(all_fields))
