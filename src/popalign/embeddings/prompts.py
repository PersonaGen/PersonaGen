# Prompts for different survey types

SPECIFIC_PROMPTS = [
    "Generate 1000 global citizens who reflect the average attitudes, beliefs, and social values held by people around the world.",
    "Generate 1000 personas representing typical global citizens in terms of how they think about society, politics, religion, and culture.",
    "Generate 1000 personas whose social trust, moral values, and political views are representative of the global human population.",
    "Generate 1000 global citizens capturing the overall cultural, social, and political norms commonly shared by people worldwide.",
    "Generate 1000 global citizens whose views on family, work, society, and government reflect the general trends observed worldwide.",
    "Generate 1000 personas representing the average human perspective on topics like community life, personal freedoms, and social responsibilities.",
    "Generate 1000 global citizens whose values and beliefs mirror the typical patterns seen in surveys of people around the world.",
    "Generate 1000 personas who reflect the mainstream attitudes and priorities of people globally, across diverse cultures and backgrounds.",
    "Generate 1000 global citizens who embody the general worldview and everyday concerns of humanity as a whole.",
    "Generate 1000 personas whose opinions and values are balanced representations of the global human population across major life domains."
]


SPECIFIC_PROMPTS_WVS_EAST = [    # China (Mainland) – two variants
    "Generate 1000 personas from China (Mainland) whose social values, cultural beliefs, and political attitudes reflect the national average across demographics.",
    "Generate 1000 personas from China (Mainland) who exemplify typical viewpoints on family life, work ethics, societal roles, and governance in modern Chinese society.",

    # Japan – two variants
    "Generate 1000 personas from Japan whose cultural norms, social beliefs, and political perspectives are representative of the Japanese population.",
    "Generate 1000 personas from Japan reflecting average attitudes toward tradition, technological innovation, and civic engagement in contemporary Japanese society.",

    # South Korea – two variants
    "Generate 1000 personas from South Korea whose values on education, community bonds, and governmental trust align with prevailing South Korean social trends.",
    "Generate 1000 personas from South Korea embodying mainstream opinions on popular culture, economic aspirations, and social cohesion in today’s South Korea.",

    # Single prompts for the remaining four
    "Generate 1000 personas from Hong Kong whose worldviews, social trust levels, and cultural identities represent the general Hong Kong populace.",
    "Generate 1000 personas from Macau whose everyday concerns, social values, and political viewpoints reflect the typical experiences of Macau residents.",
    "Generate 1000 personas from Mongolia whose cultural heritage, social beliefs, and political attitudes mirror the national distribution across both urban and rural areas.",
    "Generate 1000 personas from Taiwan (Republic of China) whose societal values, cultural traditions, and political perspectives correspond to the average Taiwanese viewpoint."
]

SPECIFIC_PROMPTS_WVS_EUROPE = [
    # Germany – four variants
    "Generate 1000 personas from Germany whose social values, cultural traditions, and political attitudes reflect the national average across demographics.",
    "Generate 1000 personas from Germany whose views on family life, work–life balance, and social welfare represent typical German perspectives.",
    "Generate 1000 personas from Germany whose opinions on environmental protection, sustainability, and economic growth mirror mainstream attitudes.",
    "Generate 1000 personas from Germany whose perspectives on migration, European integration, and national identity align with prevalent public sentiment.",

    # Netherlands – three variants
    "Generate 1000 personas from the Netherlands whose cultural norms, social tolerance, and political views are representative of Dutch society.",
    "Generate 1000 personas from the Netherlands whose attitudes toward social welfare, healthcare, and quality of life reflect common Dutch experiences.",
    "Generate 1000 personas from the Netherlands whose viewpoints on innovation, technology adoption, and community participation typify the national average.",

    # United Kingdom (Great Britain) – three variants
    "Generate 1000 personas from the United Kingdom whose societal values, regional identities, and political opinions represent the average British viewpoint.",
    "Generate 1000 personas from the United Kingdom whose perspectives on multiculturalism, urban versus rural life, and civic engagement mirror mainstream British attitudes.",
    "Generate 1000 personas from the United Kingdom whose views on the monarchy, the legacy of Brexit, and national cohesion reflect typical public sentiment."
]

SPECIFIC_PROMPTS_WVS_NORTH_AMERICA = [
    # United States – four variants
    "Generate 1000 personas from the United States whose social values, cultural beliefs, and political attitudes reflect the demographic and regional diversity of the country.",
    "Generate 1000 personas from the United States whose opinions on healthcare, education, and economic opportunity represent typical American viewpoints.",
    "Generate 1000 personas from the United States whose perspectives on national identity, civic engagement, and social justice mirror mainstream public sentiment.",
    "Generate 1000 personas from the United States whose everyday concerns, cultural preferences, and political priorities align with the average U.S. population.",

    # Canada – three variants
    "Generate 1000 personas from Canada whose social values, cultural norms, and political views represent the diversity of Canadian provinces and territories.",
    "Generate 1000 personas from Canada whose attitudes toward multiculturalism, healthcare, and environmental stewardship reflect common Canadian perspectives.",
    "Generate 1000 personas from Canada whose opinions on indigenous relations, bilingualism, and social welfare mirror the average Canadian viewpoint.",

    # Mexico – three variants
    "Generate 1000 personas from Mexico whose cultural traditions, social beliefs, and political attitudes reflect the national distribution across urban and rural regions.",
    "Generate 1000 personas from Mexico whose perspectives on family structures, economic development, and societal values represent typical Mexican viewpoints.",
    "Generate 1000 personas from Mexico whose opinions on migration, regional identity, and community cohesion mirror mainstream public sentiment."
]

SPECIFIC_PROMPTS_YOUTH = [
    "Generate 1000 adolescents whose everyday health behaviors and life experiences reflect the average patterns seen among teenagers worldwide.",
    "Generate 1000 teenage personas representing common attitudes toward relationships, friendships, family dynamics, and self-identity in contemporary youth culture.",
    "Generate 1000 young people whose diet, physical activity, and screen-time habits typify the lifestyle balance of today’s high-school–age population.",
    "Generate 1000 adolescent personas whose experiences with peer influence, risk-taking, and decision-making mirror mainstream trends among students aged 14–18.",
    "Generate 1000 teenagers whose use of substances such as vaping products, alcohol, and other drugs reflects general prevalence levels observed in youth surveys.",
    "Generate 1000 high-school–age personas capturing typical patterns in sexual development, relationship choices, and personal safety awareness.",
    "Generate 1000 adolescents whose mental-health outlook—including stress, mood, and help-seeking behavior—matches common themes reported by their peers.",
    "Generate 1000 teenage personas illustrating everyday encounters with bullying, online interactions, and feelings of belonging at school.",
    "Generate 1000 young people whose views on community engagement, future aspirations, and personal responsibility embody prevailing youth perspectives.",
    "Generate 1000 adolescents whose combined lifestyle choices, health behaviors, and social experiences present a balanced snapshot of the global teenage population."
]


WVS_SURVEY_TYPES = ("wvs", "wvs_east", "wvs_west", "wvs_north_america")
SUPPORTED_SURVEY_TYPES = WVS_SURVEY_TYPES + ("yrbss",)

SURVEY_PROMPTS = {
    "wvs": SPECIFIC_PROMPTS,
    "wvs_east": SPECIFIC_PROMPTS_WVS_EAST,
    "wvs_west": SPECIFIC_PROMPTS_WVS_EUROPE,
    "wvs_north_america": SPECIFIC_PROMPTS_WVS_NORTH_AMERICA,
    "yrbss": SPECIFIC_PROMPTS_YOUTH,
}


def get_survey_prompts(survey_type: str):
    """Return prompt templates for a supported WVS/YRBSS survey target."""
    try:
        return list(SURVEY_PROMPTS[survey_type])
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_SURVEY_TYPES)
        raise ValueError(f"Invalid survey type: {survey_type}. Use one of: {supported}.") from exc
