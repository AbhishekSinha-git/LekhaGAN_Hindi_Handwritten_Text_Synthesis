import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, features
import tqdm as tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F # <-- Added import
import math
import re
import cv2
import torch

conjunct = ['091C_094D', '0915_094D', '0924_094D']
charFolders = ['0905', '0905_0902', '0905_0903', '0906', '0907', '0908', '0909', '090A', '090F', '0910', '0913', '0914', '0915', '0915_093E', '0915_093F', '0915_0940', '0915_0941', '0915_0942', '0915_0947', '0915_0948', '0915_094B', '0915_094C', '0915_094D', '0915_094D_0937', '0915_094D_0937_0903', '0915_094D_0937_093E', '0915_094D_0937_0940', '0915_094D_0937_0941', '0915_094D_0937_0942', '0915_094D_0937_0947', '0915_094D_0937_0948', '0915_094D_0937_094B', '0915_094D_0937_094C', '0916', '0916_093E', '0916_093F', '0916_0941', '0916_0942', '0916_0948', '0916_094B', '0916_094C', '0916_094D', '0917', '0917_093E', '0917_093F', '0917_0940', '0917_0941', '0917_0942', '0917_0947', '0917_0948', '0917_094B', '0917_094C', '0917_094D', '0918', '0918_093E', '0918_093F', '0918_0940', '0918_0941', '0918_0942', '0918_0947', '0918_0948', '0918_094B', '0918_094C', '0918_094D', '0919', '0919_0902', '0919_0903', '0919_093E', '0919_093F', '0919_0940', '0919_0941', '0919_0942', '0919_0947', '0919_0948', '0919_094B', '0919_094C', '091A', '091A_0902', '091A_0903', '091A_093E', '091A_093F', '091A_0940', '091A_0941', '091A_0942', '091A_0947', '091A_0948', '091A_094B', '091A_094C', '091B', '091B_0902', '091B_0903', '091B_093E', '091B_093F', '091B_0940', '091B_0941', '091B_0942', '091B_0947', '091B_0948', '091B_094B', '091B_094C', '091C', '091C_0902', '091C_0903', '091C_093E', '091C_093F', '091C_0940', '091C_0941', '091C_0942', '091C_0947', '091C_0948', '091C_094B', '091C_094C', '091C_094D_091E', '091C_094D_091E_0902', '091C_094D_091E_0903', '091C_094D_091E_093E', '091C_094D_091E_093F', '091C_094D_091E_0940', '091C_094D_091E_0941', '091C_094D_091E_0942', '091C_094D_091E_0947', '091C_094D_091E_0948', '091C_094D_091E_094B', '091C_094D_091E_094C', '091D', '091D_0902', '091D_0903', '091D_093E', '091D_093F', '091D_0940', '091D_0941', '091D_0942', '091D_0947', '091D_0948', '091D_094B', '091D_094C', '091E', '091E_0902', '091E_0903', '091E_093E', '091E_093F', '091E_0940', '091E_0941', '091E_0942', '091E_0947', '091E_0948', '091E_094B', '091E_094C', '091F', '091F_0903', '091F_093E', '091F_093F', '091F_0940', '091F_0941', '091F_0942', '091F_0947', '091F_0948', '091F_094B', '091F_094C', '0920', '0920_0903', '0920_093E', '0920_093F', '0920_0940', '0920_0941', '0920_0942', '0920_0947', '0920_0948', '0920_094B', '0920_094C', '0921', '0921_0902', '0921_0903', '0921_093E', '0921_093F', '0921_0940', '0921_0941', '0921_0942', '0921_0947', '0921_0948', '0921_094B', '0921_094C', '0922', '0922_0902', '0922_0903', '0922_093E', '0922_093F', '0922_0940', '0922_0941', '0922_0942', '0922_0947', '0922_0948', '0922_094B', '0922_094C', '0923', '0923_0902', '0923_0903', '0923_093E', '0923_093F', '0923_0940', '0923_0941', '0923_0942', '0923_0947', '0923_0948', '0923_094B', '0923_094C', '0924', '0924_0902', '0924_0903', '0924_093E', '0924_093F', '0924_0940', '0924_0941', '0924_0942', '0924_0947', '0924_0948', '0924_094B', '0924_094C', '0924_094D_0930', '0924_094D_0930_0902', '0924_094D_0930_0903', '0924_094D_0930_093E', '0924_094D_0930_093F', '0924_094D_0930_0940', '0924_094D_0930_0941', '0924_094D_0930_0942', '0924_094D_0930_0947', '0924_094D_0930_0948', '0924_094D_0930_094B', '0924_094D_0930_094C', '0925', '0925_0902', '0925_093E', '0925_093F', '0925_0940', '0925_0941', '0925_0942', '0925_0947', '0925_0948', '0925_094B', '0925_094C', '0926', '0926_0902', '0926_0902_0903', '0926_0903', '0926_093E', '0926_093F', '0926_0940', '0926_0941', '0926_0942', '0926_0947', '0926_0948', '0926_094B', '0926_094C', '0927', '0927_0902', '0927_0903', '0927_093E', '0927_093F', '0927_0940', '0927_0941', '0927_0942', '0927_0947', '0927_0948', '0927_094B', '0927_094C', '0928', '0928_0903', '0928_093E', '0928_093F', '0928_0940', '0928_0941', '0928_0942', '0928_0947', '0928_0948', '0928_094B', '0928_094C', '092A', '092A_0902', '092A_0903', '092A_093E', '092A_093F', '092A_0940', '092A_0941', '092A_0942', '092A_0947', '092A_0948', '092A_094B', '092A_094C', '092B', '092B_0902', '092B_0903', '092B_093E', '092B_093F', '092B_0940', '092B_0941', '092B_0942', '092B_0947', '092B_0948', '092B_094B', '092B_094C', '092C', '092C_0902', '092C_0903', '092C_093E', '092C_093F', '092C_0940', '092C_0941', '092C_0942', '092C_0947', '092C_0948', '092C_094B', '092C_094C', '092D', '092D_0902', '092D_0903', '092D_093E', '092D_093F', '092D_0940', '092D_0941', '092D_0942', '092D_0947', '092D_0948', '092D_094B', '092D_094C', '092E', '092E_0902', '092E_0903', '092E_093E', '092E_093F', '092E_0940', '092E_0941', '092E_0942', '092E_0947', '092E_0948', '092E_0948_0902', '092E_094B', '092E_094C', '092F', '092F_0902', '092F_0903', '092F_093E', '092F_093F', '092F_0940', '092F_0941', '092F_0942', '092F_0947', '092F_0948', '092F_094B', '092F_094C', '0930', '0930_0902', '0930_0903', '0930_093E', '0930_093F', '0930_0940', '0930_0941', '0930_0942', '0930_0947', '0930_0948', '0930_094B', '0930_094C', '0932', '0932_0902', '0932_0903', '0932_093E', '0932_093F', '0932_0940', '0932_0941', '0932_0942', '0932_0947', '0932_0948', '0932_094B', '0932_094C', '0935', '0935_0902', '0935_0903', '0935_093E', '0935_093F', '0935_0940', '0935_0941', '0935_0942', '0935_0947', '0935_0948', '0935_094B', '0935_094C', '0936', '0936_0902', '0936_0903', '0936_093E', '0936_093F', '0936_0940', '0936_0941', '0936_0942', '0936_0947', '0936_0948', '0936_094B', '0936_094C', '0937', '0937_0902', '0937_0903', '0937_093E', '0937_093F', '0937_0940', '0937_0941', '0937_0942', '0937_0947', '0937_0948', '0937_094B', '0937_094C', '0938', '0938_0902', '0938_0903', '0938_093E', '0938_093F', '0938_0940', '0938_0941', '0938_0942', '0938_0947', '0938_0948', '0938_094B', '0938_094C', '0939', '0939_0902', '0939_0903', '0939_093E', '0939_093F', '0939_0940', '0939_0941', '0939_0942', '0939_0947', '0939_0948', '0939_094B', '0939_094C', '0966', '0967', '0968', '0969', '096A', '096B', '096C', '096D', '096E', '096F']

def getLetterTokens(word: str):

    wordComb = []
    approved_words = []

    # for each word in dict

    hindi_word = word
    charList = []

    if(len(word) <= 31): # set word length limit to 31 characters (including matras)
        word = word.replace('\n', '')
        characters = []
        for ch in word: # convert letters to unicode representations and store
            characters.append(('0' + hex(ord(ch))[2:]).upper())
        
        check = True
        i = 0
        # for each unicode character representation of current word
        while check and i < len(characters):
            check = False
            word = ''

            # add join current and next char
            if i < len(characters) - 1:
                word = characters[i] + '_' + characters[i+1]
            
            # if half (halant) letter exists in combination to next character
            if word in conjunct and i < len(characters) - 2:
                word2 = word + '_' + characters[i+2] # join with next character to check for more possibilities
                if word2 in charFolders: # if the current handwritten character combination exists
                    # if still not reached end and concatenation of next character exists in folder
                    if i < len(characters) - 3 and word2 + '_' + characters[i+3] in charFolders:
                        charList.append(word2 + '_' + characters[i+3]) # add to charlist
                        check = True
                        i += 4
                    else: # add the next character as a seperate sequence element
                        charList.append(word2)
                        check = True
                        i += 3
            
            # above if condition only adds char to charlist if that subsequence is found in folder names, and so check is set to true
            # below, if check is false, only then we fall back to the word (character[i] + character[i+1])
            # or character (character[i]) combination addition to charlist
            
            # check if word (character[i] + character[i+1]) combination exists and adds to charlist
            # and sets flag to true
            if check == False and word in charFolders:
                check = True
                charList.append(word)
                i += 2
            
            # if word also does not exist, then only the character is added to charlist given that it exists in the folder
            if check == False and characters[i] in charFolders:
                check = True
                charList.append(characters[i])
                i += 1
        
        # appends all information for that word as well as the annotated word
        if check == True:
            wordComb.append((charList, word))
            approved_words.append(hindi_word)
    
    # if(wordComb != []):
    return wordComb[0][0], approved_words
    # else:
    #     return None, None

def find_content_bounds(patch_np_uint8, background_color, threshold, smooth_window, rel_thresh, min_thresh, imgH):
    """Finds the start and end column of significant content."""
    # Use brightness threshold if background isn't perfectly uniform or 0/255
    content_mask_np = (patch_np_uint8 > threshold)
    # Or use background color if reliable:
    # content_mask_np = (patch_np_uint8 != background_color)

    column_sums = np.sum(content_mask_np, axis=0).astype(float)

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed_sums = np.convolve(column_sums, kernel, mode='same')
    else:
        smoothed_sums = column_sums

    dynamic_threshold = imgH * rel_thresh
    final_threshold = max(min_thresh, dynamic_threshold)

    indices_above_threshold = np.where(smoothed_sums > final_threshold)[0]

    first_col = -1
    last_col = -1
    if indices_above_threshold.size > 0:
        first_col = indices_above_threshold[0]
        last_col = indices_above_threshold[-1]
        # Optional buffer for end to catch faint tails
        last_col = min(last_col + 2, patch_np_uint8.shape[1] - 1)

    return first_col, last_col


# --- Modified Generation Function with Correct Stitching ---
def generate_words_autoregressive_stitched_no_overlap(
    words_list,
    generator,
    encoder,
    # ft_wv,
    noise_dim,
    imgH,
    imgW_patch,
    context_width, # Width of context Generator expects as input
    background_color,
    device,
    fixed_noise=None,
    stretch_factor=1.0,
    debug=False
):
    generator.eval()
    output_word_image_tensors = []
    output_words = []
    normalized_background = float(background_color / 255.0)

    num_words = len(words_list)

    word_iterator = tqdm(words_list, desc="Generating Words", disable=debug) if debug else words_list
    for word_idx, word_to_generate in enumerate(word_iterator):
        # 1. Tokenize & Embed
        try:
            char_unicodes, _ = getLetterTokens(word_to_generate); assert char_unicodes
            with torch.no_grad(): embeddings_np = encoder.wv[char_unicodes]; embeddings = torch.from_numpy(embeddings_np).to(device).float(); assert embeddings.shape[0] == len(char_unicodes)
        except Exception as e: print(f"Error inputs '{word_to_generate}': {e}. Skip."); continue

        # 2. Get Noise
        noise_vector = fixed_noise[word_idx:word_idx+1].to(device) if fixed_noise is not None else torch.randn(1, noise_dim).to(device)

        # 3. Autoregressive Loop
        # --- Initialize Canvas ---
        # Start with a small blank canvas (or just the first context)
        final_canvas = torch.full((1, 1, imgH, context_width), normalized_background, dtype=torch.float32).to(device)
        # -------------------------
        current_context_patch = torch.full((1, 1, imgH, imgW_patch), normalized_background, dtype=torch.float32).to(device)

        # --- Setup Debug Plotting ---
        # ... (Keep debug plot setup if debug=True) ...
        debug_fig, debug_axes = None, None
        if debug:
             num_chars = len(embeddings); fig_rows = num_chars * 2; fig_height_per_row = 1.5; fig_width = 6
             debug_fig, debug_axes = plt.subplots(fig_rows, 1, figsize=(fig_width, fig_height_per_row * fig_rows))
             if fig_rows <= 1 : debug_axes = np.array([debug_axes])
             debug_fig.suptitle(f"Debug Steps: '{word_to_generate}'", fontsize=12); debug_plot_idx = 0

        for i in range(len(embeddings)):
            char_embedding = embeddings[i:i+1]
            noise_step = noise_vector

            with torch.no_grad():
                generated_patch = generator(current_context_patch, char_embedding, noise_step) # [1,1,H,W_patch]

            # --- Find Content Bounds in Generated Patch ---
            generated_patch_np_uint8 = (generated_patch.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
            # We need the END column based on *all* content in the patch
            _, last_content_col = find_content_bounds(
                generated_patch_np_uint8, background_color, FOREGROUND_THRESHOLD,
                SMOOTHING_WINDOW_SIZE, RELATIVE_DENSITY_THRESHOLD, MIN_DENSITY_THRESHOLD, imgH
            )
            # We also need the START column *after* the input context area
            # Re-run find bounds on the *generated part only*
            generated_part_np = generated_patch_np_uint8[:, context_width:]
            first_content_col_in_gen_part, last_content_col_in_gen_part = find_content_bounds(
                generated_part_np, background_color, FOREGROUND_THRESHOLD,
                SMOOTHING_WINDOW_SIZE, RELATIVE_DENSITY_THRESHOLD, MIN_DENSITY_THRESHOLD, imgH
            )
            # Adjust first_col to be relative to the full 64x128 patch
            if first_content_col_in_gen_part != -1:
                first_content_col_overall = context_width + first_content_col_in_gen_part
            else:
                # If generated part is blank, effectively no new content starts
                first_content_col_overall = -1


            # --- Stitching: Append relevant part to canvas ---
            if first_content_col_overall != -1 and last_content_col != -1 and last_content_col >= first_content_col_overall:
                # Extract the generated content starting from first detected pixel after context
                # up to the last detected pixel overall
                content_to_append = generated_patch[:, :, :, first_content_col_overall : last_content_col + 1]
                # Concatenate to the existing canvas
                final_canvas = torch.cat((final_canvas, content_to_append.to(final_canvas.device)), dim=3)
            # If generated part was blank, canvas remains unchanged

            # --- Prepare Context for Next Step (Based on *current* generation's end) ---
            # Use last_content_col found from the full generated patch
            context_end_x_for_next = last_content_col + 1
            context_start_x_for_next = max(0, context_end_x_for_next - context_width)
            actual_context_width_for_next = context_end_x_for_next - context_start_x_for_next

            next_context_patch = torch.full_like(current_context_patch, normalized_background).to(device)
            context_data_for_display = None

            if last_content_col >= 0 and actual_context_width_for_next > 0:
                 # Extract context from the *ORIGINAL* generated patch tensor
                 context_data = generated_patch[:, :, :, context_start_x_for_next:context_end_x_for_next]
                 # Place it at the *start* of the next context patch
                 next_context_patch[:, :, :, :actual_context_width_for_next] = context_data
                 context_data_for_display = context_data.cpu() # For debug plot

            current_context_patch = next_context_patch

            # --- Debug Plotting ---
            if debug and debug_axes is not None:
                 # Plot Original Gen Patch
                 ax = debug_axes[debug_plot_idx]; ax.imshow(generated_patch_np_uint8, cmap='gray', vmin=0, vmax=255)
                 ax.set_title(f"Step {i+1} Gen (Tok: {char_unicodes[i]})", fontsize=9)
                 if first_content_col_overall != -1: ax.axvline(x=first_content_col_overall - 0.5, color='lime', linestyle=':', lw=1, label=f'Start({first_content_col_overall})')
                 if last_content_col != -1: ax.axvline(x=last_content_col + 0.5, color='r', linestyle='--', lw=1, label=f'End({last_content_col})')
                 ax.axvline(x=context_width - 0.5, color='cyan', linestyle='-', lw=0.5, label='Ctx End')
                 ax.legend(fontsize=6, loc='upper right'); ax.axis('off'); debug_plot_idx += 1
                 # Plot Next Context
                 ax = debug_axes[debug_plot_idx]; display_context_np = np.ones((imgH, imgW_patch), dtype=np.uint8) * background_color
                 if context_data_for_display is not None: context_display_np = (context_data_for_display.squeeze().numpy()*255.).astype(np.uint8); display_context_np[:, :actual_context_width_for_next] = context_display_np; ax.axvline(x=actual_context_width_for_next - 0.5, color='g', linestyle='--', lw=1)
                 ax.imshow(display_context_np, cmap='gray', vmin=0, vmax=255); ax.set_title(f"Step {i+1} Next Ctx (W: {actual_context_width_for_next})", fontsize=9); ax.axis('off'); debug_plot_idx += 1

        # --- Show Debug Plot ---
        if debug and debug_fig is not None: plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show(); plt.close(debug_fig)

        # 4. Final Processing (Remove initial context, Stretch)
        if final_canvas is None or final_canvas.shape[3] <= context_width:
             if(debug): print(f"W: Final canvas empty or too small for '{word_to_generate}'. Skip.");
             continue

        # Remove the initial blank context added at the start
        final_word_only_tensor = final_canvas[:, :, :, context_width:]

        if stretch_factor < 1.0:
            W_gen = final_word_only_tensor.shape[3]; W_final = max(W_gen, math.ceil(W_gen / stretch_factor))
            try:
                stretched_image_tensor = F.interpolate(final_word_only_tensor, size=(imgH, W_final), mode='bilinear', align_corners=False)
                stretched_image_tensor = torch.clamp(stretched_image_tensor, 0.0, 1.0)
            except Exception as e:
                stretched_image_tensor = final_word_only_tensor
                if(debug): print(f"W: Stretching failed '{word_to_generate}': {e}.");
                else: pass
        else: stretched_image_tensor = final_word_only_tensor

        target_width = 704
        current_stretched_width = stretched_image_tensor.shape[3]

        if current_stretched_width < target_width:
            # Pad on the right side
            pad_width = target_width - current_stretched_width
            # Pad with normalized background value
            final_output_tensor = F.pad(stretched_image_tensor, (0, pad_width), mode='constant', value=normalized_background)
            # print(f"Padding '{word_to_generate}' from {current_stretched_width} to {target_width}")
        elif current_stretched_width > target_width:
            # Crop from the right side
            final_output_tensor = stretched_image_tensor[:, :, :, :target_width]
            # print(f"Cropping '{word_to_generate}' from {current_stretched_width} to {target_width}")
        else:
            # Width is already correct
            final_output_tensor = stretched_image_tensor
        
        output_word_image_tensors.append(final_output_tensor.cpu()) # Store final CPU tensor
        output_words.append(word_to_generate)

    # print("Finished generating words.")
    return output_word_image_tensors, output_words

FOREGROUND_THRESHOLD = 240     # Pixels brighter than this considered foreground (adjust if background is white)
SMOOTHING_WINDOW_SIZE = 5      # Size of moving average window for column sums
RELATIVE_DENSITY_THRESHOLD = 0.05 # Column 'content' if smoothed sum > 5% of max possible
MIN_DENSITY_THRESHOLD =0        # Or require at least X pixels absolute minimum

noise_dim = 128
def generateWordImage(words, generator, encoder, device, debug=False):
    noise_dim_gen = 128            # Must match training
    imgH_gen = 64
    imgW_patch_gen = 128          # Width of the generator's OUTPUT patch
    context_width_gen = 26        # Approx 20% of 128, adjust if needed
    background_color_gen = 0      # The pixel value for blank background [0..255]

    num_words = len(words)
    example_fixed_noise = torch.randn(num_words, noise_dim).to(device)

    # print("\n--- Generating with Debug & No Overlap Stitching ---")
    generated_stitched, output_words = generate_words_autoregressive_stitched_no_overlap( # Call new function
        words_list=words, generator=generator, encoder=encoder,
        noise_dim=noise_dim, imgH=imgH_gen, imgW_patch=imgW_patch_gen, context_width=context_width_gen,
        background_color=background_color_gen, device=device, fixed_noise=None,
        stretch_factor=0.70, debug=debug
    )
    generated_stitched = [x for x in generated_stitched if x != [] or x != None]
    if(generated_stitched == None or generated_stitched == []): return generated_stitched, output_words
    else:
        generated_stitched = torch.stack(generated_stitched, dim=0).squeeze(1).squeeze(1)
        return generated_stitched, output_words

def crop_and_resize(img_array, target_height):
    if img_array is None or img_array.size == 0:
        print("Warning: Input image is empty.")
        fallback_width = int(target_height * 3) # Default aspect ratio for blank
        return np.full((target_height, fallback_width), 255, dtype=np.uint8)
    if img_array.dtype not in [np.float32, np.float64]:
         print(f"Warning: Input image dtype is {img_array.dtype}, expected float. Proceeding cautiously.")
    if np.max(img_array) > 1.0 or np.min(img_array) < 0.0:
         print("Warning: Input image values are outside the expected 0.0-1.0 range.")

    
    img_uint8 = (np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8)

    try:
        thresh_value, binarized_image = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Alternatively, use a fixed threshold if Otsu fails or you know the range:
        # thresh_value, binarized_image = cv2.threshold(img_uint8, 128, 255, cv2.THRESH_BINARY)
    except cv2.error as e:
        print(f"Error during thresholding: {e}. Returning blank image.")
        fallback_width = int(target_height * 3)
        return np.full((target_height, fallback_width), 255, dtype=np.uint8)

    # Find coordinates of the foreground pixels (white pixels in the binary mask)
    coords = np.column_stack(np.where(binarized_image > 0)) # Find where pixel value is 255

    if coords.size == 0:
        # If there is no foreground content found after thresholding
        print("Warning: No foreground content found. Returning blank image.")
        fallback_width = int(target_height * 3) # Default aspect ratio
        return np.full((target_height, fallback_width), 255, dtype=np.uint8)

    # Get the bounding box for the foreground pixels
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add a small padding (optional, prevents cutting off edges)
    padding = 2
    y_min_pad = max(0, y_min - padding)
    y_max_pad = min(img_array.shape[0] - 1, y_max + padding)
    x_min_pad = max(0, x_min - padding)
    x_max_pad = min(img_array.shape[1] - 1, x_max + padding)

    # Crop the *original* float image using the padded bounding box
    cropped_image = img_array[y_min_pad : y_max_pad + 1, x_min_pad : x_max_pad + 1]

    # Check if cropping resulted in a valid image
    if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
        print("Warning: Cropped image has zero dimension. Returning blank image.")
        fallback_width = int(target_height * 3)
        return np.full((target_height, fallback_width), 255, dtype=np.uint8)

    # --- Step 2: Resize while maintaining aspect ratio based on target height ---
    h, w = cropped_image.shape
    aspect_ratio = w / h # h should not be 0 here due to the check above

    target_width = int(target_height * aspect_ratio)
    target_width = max(1, target_width) # Ensure width is at least 1 pixel

    # Resize the cropped float image
    # cv2.resize preserves the input dtype (float in this case)
    resized_image_float = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # --- Step 3: Convert final image to uint8 (0-255) ---
    # Clip again just to be safe during float operations/resizing
    final_image_uint8 = (np.clip(resized_image_float, 0.0, 1.0) * 255).astype(np.uint8)

    return final_image_uint8

def resize_and_pad(image, target_size=(256, 256)):
    original_size = image.size
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    new_image = Image.new("L", target_size, color=255)
    top_left_x = (target_size[0] - new_size[0] + 10) // 2  # Horizontal center
    top_left_y = (target_size[1] - new_size[1]) // 2  # Vertical center
    new_image.paste(resized_image, (top_left_x, top_left_y))
    return new_image

def getWordImage(word, gen, model, DEVICE, max_height=256):
    min_height = round((max_height * 200) / 256)
    image = generateWordImage([word], gen, model, DEVICE)
    contains_matras_above_shirorekha = any(char in ['ि', 'ै', 'ी', 'े', 'ृ'] for char in word)
    image = crop_and_resize(image[0][0].numpy(), max_height if contains_matras_above_shirorekha else min_height)
    return image

def filter_hindi_letters(sentence):
    # Regular expression to match only Hindi letters and spaces
    hindi_letters_only = re.sub(r'[^\u0900-\u097F\s]+', '', sentence)
    return hindi_letters_only

def place_words_on_canvas(font_size, sentence, canvas_width, gen, model, DEVICE, debug=False):

    words = filter_hindi_letters(sentence).split()
    word_images = []
    for word in words:
        try:
            word_images.append(np.asarray(255 - getWordImage(word, gen, model, DEVICE, max_height=font_size)))
        except:
            continue

    # Start with an initial canvas height

    def setRatio(line_height, num):
        return round(line_height * (num / 256))
    

    line_height = font_size
    x_offset, y_offset = setRatio(line_height, 60), setRatio(line_height, 60)
    idx = 0
    
    canvas = Image.new('L', (canvas_width, line_height + setRatio(line_height, 120)), 255)  # 'L' mode for grayscale
    
    for word_img in word_images:
        contains_matras_above_shirorekha = any(char in ['ि', 'ै', 'ी', 'े', 'ृ'] for char in words[idx])
        word_pil = Image.fromarray(word_img)
        # if debug:
        #     word_pil = add_green_borders(word_pil)
        word_width, word_height = word_pil.size

        # Check if the word exceeds the current canvas width and needs a new line
        if x_offset + word_width > canvas_width:
            x_offset = setRatio(line_height, 60)
            y_offset += line_height + np.random.randint(setRatio(line_height, 30), setRatio(line_height, 60))

            # Dynamically expand the canvas height if the new line goes beyond the current canvas height
            if y_offset + line_height > canvas.height:
                new_height = canvas.height + line_height + np.random.randint(setRatio(line_height, 30), setRatio(line_height, 60)) + setRatio(line_height, 56)
                new_canvas = Image.new('L', (canvas_width, new_height), 255)
                new_canvas.paste(canvas, (0, 0))
                canvas = new_canvas

        # Paste the word image onto the canvas
        canvas.paste(word_pil, (x_offset, y_offset if contains_matras_above_shirorekha else y_offset + setRatio(line_height, 56)))
        x_offset += word_width + np.random.randint(setRatio(line_height, 45), setRatio(line_height, 45) + (word_width * 0.1))
        
        idx += 1
    return canvas