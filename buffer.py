# Return Only Relevant Information
        for index, preprocessed_image_info in enumerate(preprocessed_images_info):
            if len(preprocessed_image_info) > 1:
                entity_string = []
                for entity in preprocessed_image_info:
                    info = f"name={entity['entity_name']}"
                    for key, value in entity.items():
                        if key != "entity_name":
                            info += f" {key}={value}"
                    entity_string.append(info)
                    
                query_emb = self.semantic_model.encode(queries[index], convert_to_tensor=True)
                entity_emb = self.semantic_model.encode(info, convert_to_tensor=True)
                
                scores = util.cos_sim(query_emb, entity_emb)[0]
                best = scores.argmax().item()

                preprocessed_images_info[index] = preprocessed_images_info[index][best]