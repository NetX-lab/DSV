import numpy
import torch.nn as nn
import transformers
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

transformers.logging.set_verbosity_error()
import html
import random
import re
import urllib.parse as ul

import ftfy
from bs4 import BeautifulSoup

"""
Will encounter following warning:
- This IS expected if you are initializing CLIPTextModel from the checkpoint of a model trained on another task
or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing CLIPTextModel from the checkpoint of a model 
that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

https://github.com/CompVis/stable-diffusion/issues/97 
according to this issue, this warning is safe.

This is expected since the vision backbone of the CLIP model is not needed to run Stable Diffusion. 
You can safely ignore the warning, it is not an error.

This clip usage is from U-ViT and same with Stable Diffusion.
"""


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    # def __init__(self, version="openai/clip-vit-huge-patch14", device="cuda", max_length=77):
    def __init__(self, path, device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
        self.transformer = CLIPTextModel.from_pretrained(path, subfolder="text_encoder")
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        tokens = batch_encoding["input_ids"].to(self.device)

        mask = batch_encoding["attention_mask"].to(self.device)

        outputs = self.transformer(input_ids=tokens, attention_mask=mask)

        z = outputs.last_hidden_state
        pooled_z = outputs.pooler_output
        return z, pooled_z, mask

    def encode(self, text):
        return self(text)


class TextEmbedder(nn.Module):
    """
    Embeds text prompt into vector representations. Also handles text dropout for classifier-free guidance.
    """

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + "\)"
        + "\("
        + "\]"
        + "\["
        + "\}"
        + "\{"
        + "\|"
        + "\\"
        + "\/"
        + "\*"
        + r"]{1,}"
    )  # noqa

    def __init__(self, path, dropout_prob=0.1):
        super().__init__()
        self.text_encodder = FrozenCLIPEmbedder(path=path)
        self.dropout_prob = dropout_prob

    def token_drop(self, text_prompts, force_drop_ids=None):
        """
        Drops text to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = numpy.random.uniform(0, 1, len(text_prompts)) < self.dropout_prob
        else:
            # TODO
            drop_ids = force_drop_ids == 1
        labels = list(numpy.where(drop_ids, "", text_prompts))
        # print(labels)
        return labels

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption
        )

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(
            self.bad_punct_regex, r" ", caption
        )  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
        )
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
        )  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    def forward(self, text_prompts, train=False, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            text_prompts = self.token_drop(text_prompts, force_drop_ids)

        # apply the clean_caption function to the text_prompts, each item in the list could be a string or a list of strings
        text_prompts = [self.clean_caption(prompt) for prompt in text_prompts]

        # print the text_prompts in 0.05% probability
        if random.random() < 0.0005:
            print(f"text_prompts after clean_caption: {text_prompts}")

        embeddings, pooled_embeddings, mask = self.text_encodder(text_prompts)
        # return embeddings, pooled_embeddings
        return embeddings, pooled_embeddings, mask


if __name__ == "__main__":
    r"""
    Returns:

    Examples from CLIPTextModel:

    ```python
    >>> from transformers import AutoTokenizer, CLIPTextModel

    >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
    ```"""

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder = (
        TextEmbedder(path="stabilityai/stable-diffusion-2-1-base", dropout_prob=0.00001)
        .to(device)
        .to(torch.bfloat16)
    )

    text_prompt = [
        ["a photo of a cat", "a photo of a dog"],
        "a photo of a dog",
        "a photo of a dog human",
        "a-beautiful-girl in # the@@@@ world play)_with_a_cat",
    ]

    text_prompt = [
        "apply eye make up",
        "apply lipstick",
        "archery",
        "baby crawling",
        "balance beam",
        "band marching",
        "baseball pitch",
        "basketball",
        "basketball dunk",
        "bench press",
        "biking",
        "billiards",
        "blow dry hair",
        "blowing candles",
        "body weight squats",
        "bowling",
        "boxing punching bag",
        "boxing speed bag",
        "breaststroke",
        "brushing teeth",
        "clean and jerk",
        "cliff diving",
        "cricket bowling",
        "cricket shot",
        "cutting in kitchen",
        "diving",
        "drumming",
        "fencing",
        "field hockey penalty",
        "floor gymnastics",
        "frisbee catch",
        "front crawl",
        "golf swing",
        "haircut",
        "hammer throw",
        "hammering",
        "handstand pushups",
        "handstand walking",
        "head massage",
        "high jump",
        "horse race",
        "horse riding",
        "hula hoop",
        "ice dancing",
        "javelin throw",
        "juggling balls",
        "jump rope",
        "jumping jack",
        "kayaking",
        "knitting",
        "long jump",
        "lunges",
        "military parade",
        "mixing",
        "mopping floor",
        "nun chucks",
        "parallel bars",
        "pizza tossing",
        "playing cello",
        "playing daf",
        "playing dhol",
        "playing flute",
        "playing guitar",
        "playing piano",
        "playing sitar",
        "playing tabla",
        "playing violin",
        "pole vault",
        "pommel horse",
        "pullups",
        "punch",
        "pushups",
        "rafting",
        "rock climbing indoor",
        "rope climbing",
        "rowing",
        "salsa spin",
        "shaving beard",
        "shot put",
        "skateboarding",
        "skiing",
        "ski jet",
        "skydiving",
        "soccer juggling",
        "soccer penalty",
        "still rings",
        "sumo wrestling",
        "surfing",
        "swing",
        "table tennis shot",
        "taichi",
        "tennis swing",
        "throw discus",
        "trampoline jumping",
        "typing",
        "uneven bars",
        "volleyball spiking",
        "walking with dog",
        "wall pushups",
        "writing on board",
        "yoyo",
    ]
    # text_prompt = ('None', 'None', 'None')
    text_prompt = [
        "the video features a close-up of a stack of pancakes on a white plate. the pancakes are topped with a generous amount of chocolate sauce and sliced strawberries. the chocolate sauce is drizzled over the pancakes and strawberries, creating a visually appealing contrast against the white plate. the pancakes appear to be fluffy and light, while the chocolate sauce has a glossy sheen. the strawberries are fresh and bright red, adding a pop of color to the dish. the background is blurred, but it appears to be a kitchen setting with a plant and a window.",
        "the video shows a scene from a video game where a large dragon is flying in the sky. the dragon is being attacked by a group of people on the ground who are using weapons to fight it. the dragon is breathing fire and the people are trying to dodge the flames. the scene is set in a dark and stormy environment, and the dragon is the main focus of the video.",
        "in the video, a person is seen working on the interior of a car. the person is holding a wire and appears to be plugging it into a connector in the car. the car has a red exterior and a black interior. the person is wearing a black jacket and a watch on their left wrist. the video is shot from a side angle, and the person's actions are focused on the wiring process.",
        "in the video, we see a woman holding a small pastry in her hand. she takes a bite out of it and then shows the inside of the pastry to the camera. the woman seems to be enjoying the pastry as she takes another bite. the pastry appears to be a small, golden-brown, flaky pastry, possibly a croissant or a similar type of pastry. the woman is wearing a black and gold dress and has red lipstick on. the background is a living room with a couch and a picture frame on the wall.",
    ]
    output = text_encoder(text_prompts=text_prompt, train=False)
    # print(output)
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    # print(output.shape)
    print(output[2])
